#!/usr/bin/env python3
"""step 7: stage-2 acoustic-unfrozen sft (hypotheses b + c + speaking-rate data filter).

combines:
1. speaking-rate (chars/sec) label-noise filter across all 286k training clips.
2. hypothesis b: merges step-6 sft into base weights, then unfreezes 100% of the multimodal
   audio projector (`embed_audio`) + top 4 conformer layers of `audio_tower` in full-rank
   bfloat16 alongside rank-16 lora on the llm decoder.
3. hypothesis c: zero label smoothing (0.0) + 3.0x loss weight on the `<end_of_turn>` stop
   token for clips < 2.0s so the decoder stops immediately on 1-word backchannels.
4. built-in 180-clip live validation evaluator with strict cer/wer save gate & early stopping.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from jiwer import cer as corpus_cer, wer as corpus_wer
from torch.utils.data import DataLoader, Dataset, Sampler
import transformers
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    LogitsProcessorList,
    get_cosine_schedule_with_warmup,
)

from gu_normalizer import compute_utterance_metrics, normalize_gujarati
from run_baselines_optimized import (
    DATA_CACHE,
    RESULTS_DIR,
    TARGET_SR,
    _cast_audio_columns_without_decoding,
    _decode_audio_value,
    _to_mono_16k,
    hf_auth_token,
    load_gujarati_dataset,
)
from step5_gemma_zero import (
    Gemma4Step5Adapter,
    GujaratiScriptLogitsProcessor,
    build_banned_script_token_tensor,
)

transformers.logging.set_verbosity_error()

WORKSPACE = Path(__file__).resolve().parent
SFT_CHECKPOINT_DIR = WORKSPACE / "checkpoints" / "gemma4_step6_sft"
GRPO_CHECKPOINT_DIR = WORKSPACE / "checkpoints" / "gemma4_step7_grpo"
AUDIO_TOP_STATE_NAME = "unfrozen_audio_top.pt"
PROMPT = (
    "Transcribe the following speech segment in Gujarati script. "
    "Output only the verbatim Gujarati transcription, with no explanation."
)


def _prepare_ds_no_decode(ds: Any, default_audio_col: str) -> tuple[Any, str]:
    res = _cast_audio_columns_without_decoding(ds)
    if isinstance(res, tuple):
        ds_obj = res[0]
        detected_col = res[1] if (len(res) > 1 and isinstance(res[1], str)) else None
    else:
        ds_obj = res
        detected_col = None

    cols = list(getattr(ds_obj, "column_names", []) or [])
    if detected_col and detected_col in cols:
        audio_col = detected_col
    elif default_audio_col in cols:
        audio_col = default_audio_col
    elif "audio_filepath" in cols:
        audio_col = "audio_filepath"
    elif "audio" in cols:
        audio_col = "audio"
    else:
        audio_col = cols[0]
    return ds_obj, audio_col


def is_clean_speaking_rate(norm_text: str, duration_s: float) -> bool:
    """rejects untranscribed long audio (<3.5 chars/s) and truncated short audio (>21 chars/s)."""
    if not norm_text or duration_s < 0.32 or duration_s > 20.0:
        return False
    n_chars = len(norm_text)
    n_words = len(norm_text.split())
    cps = n_chars / duration_s

    if duration_s < 1.5:
        return (1 <= n_words <= 3) and (2 <= n_chars <= 22) and (2.5 <= cps <= 18.0)
    return (n_words >= 2) and (4.0 <= cps <= 21.0)


class Stage2CleanedDataset(Dataset):
    def __init__(
        self,
        max_short_iv: int = 1200,
        max_fleurs_repeats: int = 2,
        max_kathbath: int = 12000,
        max_regular_iv: int = 12000,
        specaugment: bool = True,
        seed: int = 2026,
    ):
        super().__init__()
        token = hf_auth_token()
        rng = random.Random(seed)
        self.specaugment = specaugment
        self.datasets: dict[str, tuple[Any, str, str]] = {}
        self.records: list[dict[str, Any]] = []
        rejected_noise = 0

        print("[Stage2 Data] Indexing & speaking-rate filtering IndicVoices Gujarati train...")
        ds_iv = load_dataset(
            "ai4bharat/IndicVoices", "gujarati", split="train", cache_dir=str(DATA_CACHE), token=token
        )
        ds_iv, iv_audio_col = _prepare_ds_no_decode(ds_iv, "audio_filepath")
        iv_text_col = "text" if "text" in ds_iv.column_names else "normalized"
        self.datasets["indicvoices"] = (ds_iv, iv_audio_col, iv_text_col)

        iv_durs = ds_iv["duration"] if "duration" in ds_iv.column_names else [3.0] * len(ds_iv)
        iv_texts = ds_iv[iv_text_col]
        short_iv_pool, reg_iv_pool = [], []
        for idx, (dur, txt) in enumerate(zip(iv_durs, iv_texts)):
            d = float(dur or 0.0)
            norm_t = normalize_gujarati(str(txt or ""))
            if not is_clean_speaking_rate(norm_t, d):
                rejected_noise += 1
                continue
            if d < 1.5:
                short_iv_pool.append({"ds": "indicvoices", "idx": idx, "duration": d, "text": norm_t})
            else:
                reg_iv_pool.append({"ds": "indicvoices", "idx": idx, "duration": d, "text": norm_t})

        rng.shuffle(short_iv_pool)
        rng.shuffle(reg_iv_pool)
        short_selected = short_iv_pool[:max_short_iv]
        reg_selected = reg_iv_pool[:max_regular_iv]

        print("[Stage2 Data] Indexing FLEURS Gujarati train...")
        fleurs_pq = hf_hub_download(
            repo_id="google/fleurs",
            repo_type="dataset",
            filename="gu_in/train-00000-of-00001.parquet",
            revision="fdd74f7cdf01fc7641ab3072b6a38d364b3667f6",
            cache_dir=str(DATA_CACHE),
            token=token,
        )
        ds_fl = load_dataset("parquet", data_files={"train": fleurs_pq}, split="train", cache_dir=str(DATA_CACHE))
        ds_fl, fl_audio_col = _prepare_ds_no_decode(ds_fl, "audio")
        fl_text_col = "transcription" if "transcription" in ds_fl.column_names else "raw_transcription"
        self.datasets["fleurs"] = (ds_fl, fl_audio_col, fl_text_col)

        fl_texts = ds_fl[fl_text_col]
        fl_nums = ds_fl["num_samples"] if "num_samples" in ds_fl.column_names else [16000 * 6] * len(ds_fl)
        fl_pool = []
        for idx, (n_samp, txt) in enumerate(zip(fl_nums, fl_texts)):
            d = float(n_samp or 96000) / 16000.0 if float(n_samp or 0) > 100 else 6.0
            norm_t = normalize_gujarati(str(txt or ""))
            if is_clean_speaking_rate(norm_t, d):
                fl_pool.append({"ds": "fleurs", "idx": idx, "duration": d, "text": norm_t})
            else:
                rejected_noise += 1
        fl_selected = fl_pool * max(1, max_fleurs_repeats)
        rng.shuffle(fl_selected)

        print("[Stage2 Data] Indexing & speaking-rate filtering Kathbath Gujarati train...")
        ds_kb = load_dataset(
            "ai4bharat/Kathbath", "gujarati", split="train", cache_dir=str(DATA_CACHE), token=token
        )
        ds_kb, kb_audio_col = _prepare_ds_no_decode(ds_kb, "audio_filepath")
        kb_text_col = "text" if "text" in ds_kb.column_names else "transcription"
        self.datasets["kathbath"] = (ds_kb, kb_audio_col, kb_text_col)

        kb_durs = ds_kb["duration"] if "duration" in ds_kb.column_names else [5.0] * len(ds_kb)
        kb_texts = ds_kb[kb_text_col]
        kb_pool = []
        for idx, (dur, txt) in enumerate(zip(kb_durs, kb_texts)):
            d = float(dur or 5.0)
            if d > 100.0:
                d = d / 16000.0
            norm_t = normalize_gujarati(str(txt or ""))
            if is_clean_speaking_rate(norm_t, d):
                kb_pool.append({"ds": "kathbath", "idx": idx, "duration": d, "text": norm_t})
            else:
                rejected_noise += 1
        rng.shuffle(kb_pool)
        kb_selected = kb_pool[:max_kathbath]

        self.records = short_selected + reg_selected + fl_selected + kb_selected
        total_h = sum(r["duration"] for r in self.records) / 3600.0
        print(
            f"[Stage2 Data] Ready: {len(self.records):,} clean clips ({total_h:.1f} hrs) | "
            f"Rejected {rejected_noise:,} corrupted speaking-rate outliers!"
        )

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _apply_waveform_specaugment(audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        if n < int(2.5 * TARGET_SR):
            return audio
        out = audio.copy()
        out *= random.uniform(0.85, 1.15)
        for _ in range(random.randint(1, 2)):
            mask_len = random.randint(int(0.030 * TARGET_SR), int(0.055 * TARGET_SR))
            start = random.randint(int(0.2 * TARGET_SR), max(int(0.2 * TARGET_SR) + 1, n - mask_len - int(0.2 * TARGET_SR)))
            out[start : start + mask_len] *= 0.02
        return out

    def __getitem__(self, i: int) -> dict[str, Any]:
        for attempt in range(5):
            rec = self.records[(i + attempt) % len(self.records)]
            ds, audio_col, _ = self.datasets[rec["ds"]]
            try:
                row = ds[int(rec["idx"])]
                decoded = _decode_audio_value(row[audio_col])
                if isinstance(decoded, tuple):
                    audio = _to_mono_16k(decoded[0], int(decoded[1]))
                else:
                    audio = np.asarray(decoded, dtype=np.float32)
                max_len = int(20.0 * TARGET_SR)
                if len(audio) > max_len:
                    audio = audio[:max_len]
                dur = len(audio) / float(TARGET_SR)
                if not is_clean_speaking_rate(rec["text"], dur):
                    continue
                if self.specaugment:
                    audio = self._apply_waveform_specaugment(audio)
                return {"audio": audio, "duration": dur, "text": rec["text"]}
            except Exception:
                continue
        return self.__getitem__((i + 7) % len(self.records))


class DurationBucketSampler(Sampler[list[int]]):
    def __init__(self, durations: Sequence[float], batch_size: int, seed: int = 2026):
        self.durations = list(durations)
        self.batch_size = batch_size
        self.bucket_size = batch_size * 32
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.durations)))
        rng.shuffle(indices)
        batches = []
        for s in range(0, len(indices), self.bucket_size):
            bucket = indices[s : s + self.bucket_size]
            bucket.sort(key=lambda idx: self.durations[idx])
            for b in range(0, len(bucket), self.batch_size):
                batch = bucket[b : b + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self.durations) // self.batch_size


class Gemma4Stage2Collator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "right"

        full_convs, prompt_convs = [], []
        durations = []
        for f in features:
            durations.append(float(f["duration"]))
            u_msg = {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": f["audio"], "sampling_rate": TARGET_SR},
                    {"type": "text", "text": PROMPT},
                ],
            }
            a_msg = {"role": "assistant", "content": [{"type": "text", "text": f["text"]}]}
            full_convs.append([u_msg, a_msg])
            prompt_convs.append([u_msg])

        batch = self.processor.apply_chat_template(
            full_convs, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt", padding=True
        )
        prompt_batch = self.processor.apply_chat_template(
            prompt_convs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        prompt_attn = prompt_batch["attention_mask"]

        for i in range(len(features)):
            p_len = int(prompt_attn[i].sum().item())
            labels[i, :p_len] = -100
            labels[i, batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        batch["clip_durations"] = torch.tensor(durations, dtype=torch.float32)
        return batch


class OnlineValidationEvaluator:
    def __init__(self, samples_per_dataset: int = 60):
        print(f"[Live Evaluator] Pre-loading {samples_per_dataset * 3} stratified validation clips into RAM...")
        self.clips: list[dict[str, Any]] = []
        rng = random.Random(2026)

        for ds_key in ("fleurs", "indicvoices", "kathbath"):
            ds, _split_name, audio_col, text_col = load_gujarati_dataset(ds_key)
            all_indices = list(range(len(ds)))
            rng.shuffle(all_indices)
            count = 0
            for idx in all_indices:
                if count >= samples_per_dataset:
                    break
                try:
                    row = ds[int(idx)]
                    arr, sr = _decode_audio_value(row[audio_col])
                    audio = _to_mono_16k(arr, sr)
                    dur = len(audio) / float(TARGET_SR)
                    ref = str(row.get(text_col, "") or "").strip()
                    if not ref or dur < 0.30 or dur > 22.0:
                        continue
                    self.clips.append({
                        "dataset": ds_key,
                        "duration": round(dur, 3),
                        "audio": audio,
                        "ref_norm": normalize_gujarati(ref),
                    })
                    count += 1
                except Exception:
                    continue

        self.clips.sort(key=lambda x: x["duration"])
        n_short = sum(1 for c in self.clips if c["duration"] < 1.5)
        print(f"[Live Evaluator] Ready: {len(self.clips)} validation clips cached in RAM ({n_short} short <1.5s clips).")

    @torch.inference_mode()
    def evaluate(
        self,
        model: PeftModel,
        processor: Any,
        logits_processors: LogitsProcessorList,
        batch_size: int = 32,
    ) -> dict[str, float]:
        t0 = time.perf_counter()
        model.eval()
        model.config.use_cache = True
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"

        refs, hyps, capped_cers, short_cers = [], [], [], []
        for start in range(0, len(self.clips), batch_size):
            batch = self.clips[start : start + batch_size]
            convs = [
                [{
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": c["audio"], "sampling_rate": TARGET_SR},
                        {"type": "text", "text": PROMPT},
                    ],
                }]
                for c in batch
            ]
            inputs = processor.apply_chat_template(
                convs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True
            ).to(model.device)

            max_dur = max(c["duration"] for c in batch)
            max_new = min(256, max(10, int(max_dur * 20) + 4))
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                no_repeat_ngram_size=5,
                logits_processor=logits_processors,
            )
            p_len = inputs["input_ids"].shape[1]
            decoded = processor.batch_decode(gen_ids[:, p_len:], skip_special_tokens=True)

            for c, hyp_raw in zip(batch, decoded):
                m = compute_utterance_metrics(c["ref_norm"], hyp_raw)
                refs.append(c["ref_norm"])
                hyps.append(m["hyp_norm"])
                capped_cers.append(m["capped_cer"])
                if c["duration"] < 1.5:
                    short_cers.append(m["capped_cer"])

        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "right"

        val_cer = 100.0 * corpus_cer(refs, hyps)
        val_wer = 100.0 * corpus_wer(refs, hyps)
        val_capped = float(np.mean(capped_cers))
        val_short = float(np.mean(short_cers)) if short_cers else val_capped
        primary_score = 0.65 * val_cer + 0.35 * val_wer
        return {
            "val_cer": round(val_cer, 3),
            "val_wer": round(val_wer, 3),
            "val_capped_cer": round(val_capped, 3),
            "val_short_cer": round(val_short, 3),
            "primary_score": round(primary_score, 4),
            "eval_s": round(time.perf_counter() - t0, 1),
        }


def compute_eos_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    clip_durations: torch.Tensor,
    short_eos_weight: float = 3.0,
) -> torch.Tensor:
    """hypothesis c: zero label smoothing + 3.0x weight on the final <end_of_turn> token for clips < 2.0s."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []

    for i in range(shift_logits.size(0)):
        valid_positions = torch.nonzero(shift_labels[i] != -100, as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            continue
        token_nll = F.cross_entropy(
            shift_logits[i][valid_positions].float(),
            shift_labels[i][valid_positions],
            reduction="none",
        )
        weights = torch.ones_like(token_nll)
        if float(clip_durations[i].item()) < 2.0:
            # boost the final stop token (<end_of_turn>) by 3.0x on short clips
            weights[-1] = short_eos_weight
        losses.append((token_nll * weights).sum() / weights.sum())

    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def collect_llm_only_linear_targets(model: torch.nn.Module) -> list[str]:
    """collects exact torch.nn.Linear leaf layers in the llm decoder (excluding audio/vision towers)."""
    target_keywords = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    names = []
    for name, module in model.named_modules():
        lower = name.lower()
        if isinstance(module, torch.nn.Linear) and any(k in name for k in target_keywords):
            if "audio" not in lower and "vision" not in lower and "embed_audio" not in lower:
                names.append(name)
    return names


def unfreeze_audio_projector_and_top_layers(model: torch.nn.Module, top_k_layers: int = 4) -> list[str]:
    """hypothesis b: unfreezes 100% of multimodal audio projector + top K conformer layers in bfloat16."""
    # detect highest layer indices inside audio_tower
    layer_indices = set()
    for name, _ in model.named_parameters():
        if "audio_tower" in name and ".layers." in name:
            try:
                idx = int(name.split(".layers.")[1].split(".")[0])
                layer_indices.add(idx)
            except Exception:
                pass
    sorted_layers = sorted(layer_indices)
    top_layers = set(sorted_layers[-top_k_layers:]) if sorted_layers else set()

    unfrozen_names = []
    for name, param in model.named_parameters():
        lower = name.lower()
        is_proj = "embed_audio" in lower or "audio_projector" in lower or "multi_modal_projector" in lower
        is_top_audio = False
        if "audio_tower" in lower and ".layers." in lower:
            try:
                idx = int(name.split(".layers.")[1].split(".")[0])
                if idx in top_layers:
                    is_top_audio = True
            except Exception:
                pass
        if is_proj or is_top_audio:
            param.requires_grad = True
            unfrozen_names.append(name)
    return unfrozen_names


def save_stage2_checkpoint(
    model: PeftModel,
    processor: Any,
    unfrozen_audio_names: list[str],
    out_dir: Path,
) -> None:
    tmp_dir = out_dir.parent / (out_dir.name + "_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(tmp_dir))
    processor.save_pretrained(str(tmp_dir))

    # also save the full-rank unfrozen audio projector + top conformer weights
    state_dict = model.state_dict()
    unfrozen_Set = set(unfrozen_audio_names)
    audio_top_state = {k: v.detach().cpu() for k, v in state_dict.items() if k in unfrozen_Set}
    torch.save(audio_top_state, tmp_dir / AUDIO_TOP_STATE_NAME)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)


def train_stage2(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not SFT_CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Missing Step 6 SFT checkpoint: {SFT_CHECKPOINT_DIR}")

    evaluator = OnlineValidationEvaluator(samples_per_dataset=args.eval_samples_per_ds)

    dataset = Stage2CleanedDataset(
        max_short_iv=args.max_short_iv,
        max_fleurs_repeats=args.max_fleurs_repeats,
        max_kathbath=args.max_kathbath,
        max_regular_iv=args.max_regular_iv,
        specaugment=True,
    )
    durations = [r["duration"] for r in dataset.records]
    sampler = DurationBucketSampler(durations, batch_size=args.batch_size)

    token = hf_auth_token()
    print("[Stage2 Model] Loading Base Model, merging Step-6 SFT, and unfreezing Audio Projector + Top 4 Audio Layers...")
    processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it", token=token)
    base_model = AutoModelForMultimodalLM.from_pretrained(
        "google/gemma-4-E4B-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )
    # Merge Step-6 SFT weights into the base model so we start at 6.44% Macro CER
    sft_peft = PeftModel.from_pretrained(base_model, str(SFT_CHECKPOINT_DIR))
    merged_base = sft_peft.merge_and_unload()

    merged_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    merged_base.enable_input_require_grads()

    # Attach fresh LoRA to LLM decoder + unfreeze full-rank Audio Projector & Top 4 Audio Conformer layers
    llm_targets = collect_llm_only_linear_targets(merged_base)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=llm_targets,
    )
    model = get_peft_model(merged_base, lora_cfg)
    unfrozen_audio_names = unfreeze_audio_projector_and_top_layers(model, top_k_layers=4)
    model.print_trainable_parameters()
    print(f"[Hypothesis B] Unfroze {len(unfrozen_audio_names)} full-rank Audio Projector & Top-4 Conformer tensors!")

    banned_tensor = build_banned_script_token_tensor(processor.tokenizer, model, model.device)
    logits_processors = LogitsProcessorList([GujaratiScriptLogitsProcessor(banned_tensor)])

    # Save Step-0 baseline and run initial validation
    save_stage2_checkpoint(model, processor, unfrozen_audio_names, GRPO_CHECKPOINT_DIR)
    base_eval = evaluator.evaluate(model, processor, logits_processors)
    best_primary = base_eval["primary_score"]
    print(
        f"\n[Step 0 SFT Baseline Eval] val_cer={base_eval['val_cer']:.2f}% | "
        f"val_wer={base_eval['val_wer']:.2f}% | val_capped={base_eval['val_capped_cer']:.2f}% | "
        f"short_<1.5s_cer={base_eval['val_short_cer']:.2f}% | primary_score={best_primary:.3f} ({base_eval['eval_s']}s)"
    )

    unfrozen_set = set(unfrozen_audio_names)
    audio_params, llm_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in unfrozen_set or "audio" in name.lower():
            audio_params.append(p)
        else:
            llm_params.append(p)

    collator = Gemma4Stage2Collator(processor)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )

    total_opt_steps = args.max_steps if args.max_steps > 0 else max(1, len(loader) // args.grad_accum)
    optimizer = torch.optim.AdamW(
        [
            {"params": audio_params, "lr": args.audio_lr, "weight_decay": 0.01},
            {"params": llm_params, "lr": args.llm_lr, "weight_decay": 0.01},
        ],
        fused=torch.cuda.is_available(),
    )
    warmup_steps = max(10, int(0.08 * total_opt_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    trainable_params = audio_params + llm_params
    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)

    opt_step = 0
    no_improve_count = 0
    running_loss = 0.0
    micro_since_log = 0
    total_clips = 0
    t_start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "step7_stage2_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    for micro_step, batch in enumerate(loader, 1):
        clip_durations = batch.pop("clip_durations").to(model.device, non_blocking=True)
        batch_ = {k: (v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        labels = batch_.pop("labels")
        outputs = model(**batch_)
        loss = compute_eos_weighted_loss(outputs.logits, labels, clip_durations, short_eos_weight=3.0)
        (loss / args.grad_accum).backward()

        running_loss += float(loss.detach().item())
        micro_since_log += 1
        total_clips += len(batch["input_ids"])
        del outputs, batch_, labels, clip_durations

        if micro_step % args.grad_accum == 0 or micro_step == len(loader):
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            if opt_step % args.log_every == 0 or opt_step == 1 or opt_step == total_opt_steps:
                elapsed_s = time.perf_counter() - t_start
                avg_loss = running_loss / max(1, micro_since_log)
                running_loss = 0.0
                micro_since_log = 0
                vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                lrs = scheduler.get_last_lr()
                clips_s = total_clips / max(elapsed_s, 1e-6)
                print(
                    f"[Stage2 Step {opt_step:4d}/{total_opt_steps}] "
                    f"loss={avg_loss:.4f} | audio_lr={lrs[0]:.2e} | llm_lr={lrs[1]:.2e} | "
                    f"speed={clips_s:.2f} clips/s | vram={vram_gb:.1f} GB | elapsed={elapsed_s/60:.1f}m"
                )

            if opt_step % args.eval_every == 0 or opt_step == total_opt_steps:
                ev = evaluator.evaluate(model, processor, logits_processors)
                model.train()
                model.config.use_cache = False

                improved = (ev["primary_score"] < best_primary) and (ev["val_cer"] <= base_eval["val_cer"])
                status = "NEW BEST - SAVED!" if improved else f"no improvement ({no_improve_count + 1}/{args.patience})"
                print(
                    f"  >>> [LIVE EVAL Step {opt_step:4d}] "
                    f"val_cer={ev['val_cer']:.2f}% (base={base_eval['val_cer']:.2f}%) | "
                    f"val_wer={ev['val_wer']:.2f}% (base={base_eval['val_wer']:.2f}%) | "
                    f"capped={ev['val_capped_cer']:.2f}% | short_<1.5s={ev['val_short_cer']:.2f}% | "
                    f"primary={ev['primary_score']:.3f} [{status}] ({ev['eval_s']}s)"
                )
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": opt_step, **ev, "improved": improved}) + "\n")

                if improved:
                    best_primary = ev["primary_score"]
                    no_improve_count = 0
                    save_stage2_checkpoint(model, processor, unfrozen_audio_names, GRPO_CHECKPOINT_DIR)
                else:
                    no_improve_count += 1
                    if no_improve_count >= args.patience:
                        print(f"[Early Stopping] Validation primary score plateaued for {args.patience} checks. Stopping cleanly.")
                        break

            if args.max_steps > 0 and opt_step >= args.max_steps:
                break

    total_m = (time.perf_counter() - t_start) / 60.0
    print(f"\n[Stage2 Done] Best validated checkpoint preserved in {GRPO_CHECKPOINT_DIR} ({total_m:.1f} minutes).")


# =====================================================================
# INFERENCE ADAPTER FOR EVALUATION
# =====================================================================
class Gemma4GRPOAdapter(Gemma4Step5Adapter):
    """loads step-6 sft + stage-2 unfrozen audio top layers + stage-2 llm lora."""

    batch_mode = "multimodal_grpo_merged_batch"

    def __init__(self, checkpoint_dir: Path = GRPO_CHECKPOINT_DIR):
        self.name = "gemma-4-E4B-grpo"
        adapter_dir = checkpoint_dir
        if not (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "default" / "adapter_config.json").exists():
            adapter_dir = adapter_dir / "default"

        print(f"[Model] Loading {self.name} (Step-6 SFT + Stage-2 Unfrozen Audio Tower + Stage-2 LoRA)...")
        token = hf_auth_token()
        self.processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it", token=token)
        base_model = AutoModelForMultimodalLM.from_pretrained(
            "google/gemma-4-E4B-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
        # First merge Step-6 SFT if Stage-2 was trained on top of merged Step-6 SFT
        audio_top_path = adapter_dir / AUDIO_TOP_STATE_NAME
        if audio_top_path.exists() and SFT_CHECKPOINT_DIR.exists():
            sft_peft = PeftModel.from_pretrained(base_model, str(SFT_CHECKPOINT_DIR))
            base_model = sft_peft.merge_and_unload()

        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        if audio_top_path.exists():
            audio_state = torch.load(str(audio_top_path), map_location=peft_model.device)
            peft_model.load_state_dict(audio_state, strict=False)
            print(f"[Model] Loaded {len(audio_state)} full-rank unfrozen audio projector/encoder tensors.")

        self.model = peft_model.merge_and_unload().eval()

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            banned_tensor = build_banned_script_token_tensor(tokenizer, self.model, self.model.device)
            self.logits_processors = LogitsProcessorList([GujaratiScriptLogitsProcessor(banned_tensor)])
        else:
            self.logits_processors = LogitsProcessorList()

    @staticmethod
    def _max_tokens_for_chunks(chunks: Sequence[np.ndarray]) -> int:
        max_dur_s = max((len(c) / TARGET_SR for c in chunks), default=10.0)
        return min(448, max(10, int(max_dur_s * 20) + 4))

    def transcribe_batch(self, audio_arrays, row_indices, batch_size):
        refs: list[tuple[int, int]] = []
        chunks: list[np.ndarray] = []
        per_sample: list[list[str | None]] = []
        errors: list[Exception | None] = [None] * len(audio_arrays)

        for sample_i, audio in enumerate(audio_arrays):
            sample_chunks = self._split_audio_at_silence(audio)
            per_sample.append([None] * len(sample_chunks))
            for chunk_i, chunk in enumerate(sample_chunks):
                refs.append((sample_i, chunk_i))
                chunks.append(chunk)

        chunk_batch_size = max(1, min(batch_size, 32))
        for start in range(0, len(chunks), chunk_batch_size):
            chunk_group = chunks[start : start + chunk_batch_size]
            ref_group = refs[start : start + chunk_batch_size]
            try:
                texts = self._transcribe_chunks_batch(chunk_group)
                for (sample_i, chunk_i), text in zip(ref_group, texts):
                    per_sample[sample_i][chunk_i] = text
            except Exception as batch_exc:
                print(f"\n[WARN] Gemma Stage2 batch failed; retrying individually: {batch_exc!r}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for (sample_i, chunk_i), chunk in zip(ref_group, chunk_group):
                    try:
                        per_sample[sample_i][chunk_i] = self._transcribe_chunk(chunk)
                    except Exception as exc:
                        errors[sample_i] = errors[sample_i] or exc

        outputs: list[str | Exception] = []
        for sample_i, parts in enumerate(per_sample):
            if errors[sample_i] is not None:
                outputs.append(errors[sample_i])
            elif any(part is None for part in parts):
                outputs.append(RuntimeError("Gemma Stage2 did not produce every audio chunk"))
            else:
                outputs.append(self._join_overlapping_transcripts([str(p) for p in parts]))
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-short-iv", type=int, default=1200)
    parser.add_argument("--max-fleurs-repeats", type=int, default=2)
    parser.add_argument("--max-kathbath", type=int, default=12000)
    parser.add_argument("--max-regular-iv", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--audio-lr", type=float, default=1.5e-5, help="full-rank LR for audio projector & top 4 layers")
    parser.add_argument("--llm-lr", type=float, default=8.0e-6, help="gentle LoRA LR for LLM decoder")
    parser.add_argument("--eval-samples-per-ds", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=80)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    train_stage2(args)
