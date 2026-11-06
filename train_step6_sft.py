#!/usr/bin/env python3
"""step 6: high-throughput supervised fine-tuning (sft) for gemma 4 e4b gujarati asr."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn  # FIXED: Added missing import for nn.Linear
from torch.utils.data import Dataset, DataLoader, Sampler
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
from transformers.models.gemma4 import modeling_gemma4  # FIXED: Added missing import for monkey-patch

from gu_normalizer import normalize_gujarati
from run_baselines_optimized import (
    DATA_CACHE,
    RESULTS_DIR,
    TARGET_SR,
    _cast_audio_columns_without_decoding,
    _decode_audio_value,
    _to_mono_16k,
    hf_auth_token,
)
from step5_gemma_zero import (
    Gemma4Step5Adapter,
    GujaratiScriptLogitsProcessor,
    build_banned_script_token_tensor,
)

transformers.logging.set_verbosity_error()

WORKSPACE = Path(__file__).resolve().parent
CHECKPOINT_DIR = WORKSPACE / "checkpoints" / "gemma4_step6_sft"
PROMPT = (
    "Transcribe the following speech segment in Gujarati script. "
    "Output only the verbatim Gujarati transcription, with no explanation."
)


# =====================================================================
# 1. DATA LOADING & FILTERING
# =====================================================================
class CuratedGujaratiSFTDataset(Dataset):
    """lazy-decoded, metadata-filtered balanced sft dataset across fleurs, kathbath, and indicvoices."""

    def __init__(
        self,
        max_fleurs: int = 3145,
        max_kathbath: int = 10000,
        max_indicvoices: int = 12000,
        min_dur: float = 1.5,
        max_dur: float = 24.0,
        seed: int = 42,
    ):
        super().__init__()
        token = hf_auth_token()
        rng = random.Random(seed)
        self.datasets: dict[str, Any] = {}
        self.records: list[dict[str, Any]] = []

        # Load FLEURS train split
        print("[SFT Data] Indexing FLEURS Gujarati train...")
        fleurs_pq = hf_hub_download(
            repo_id="google/fleurs",
            repo_type="dataset",
            filename="gu_in/train-00000-of-00001.parquet",
            revision="fdd74f7cdf01fc7641ab3072b6a38d364b3667f6",
            cache_dir=str(DATA_CACHE),
            token=token,
        )
        ds_fl = load_dataset("parquet", data_files={"train": fleurs_pq}, split="train", cache_dir=str(DATA_CACHE))
        ds_fl, _ = _cast_audio_columns_without_decoding(ds_fl)
        self.datasets["fleurs"] = ds_fl

        fl_nums = ds_fl["num_samples"]
        fl_texts = ds_fl["transcription"]
        fl_indices = []
        for idx, (ns, txt) in enumerate(zip(fl_nums, fl_texts)):
            dur = float(ns or 0) / TARGET_SR if ns else 8.0
            norm_t = normalize_gujarati(txt)
            if min_dur <= dur <= max_dur and len(norm_t.split()) >= 2:
                fl_indices.append((idx, dur, norm_t))
        rng.shuffle(fl_indices)
        for idx, dur, norm_t in fl_indices[:max_fleurs]:
            self.records.append({"source": "fleurs", "idx": idx, "duration": dur, "text": norm_t, "audio_col": "audio"})

        # Load Kathbath train split
        if max_kathbath > 0:
            print("[SFT Data] Indexing Kathbath Gujarati train via metadata columns...")
            ds_kb = load_dataset("ai4bharat/Kathbath", "gujarati", split="train", cache_dir=str(DATA_CACHE), token=token)
            ds_kb, _ = _cast_audio_columns_without_decoding(ds_kb)
            self.datasets["kathbath"] = ds_kb

            kb_durs = ds_kb["duration"]
            kb_texts = ds_kb["text"]
            kb_candidates = [
                (idx, float(d), txt)
                for idx, (d, txt) in enumerate(zip(kb_durs, kb_texts))
                if d is not None and min_dur <= float(d) <= max_dur and txt and len(str(txt).split()) >= 2
            ]
            rng.shuffle(kb_candidates)
            for idx, dur, txt in kb_candidates[:max_kathbath]:
                norm_t = normalize_gujarati(txt)
                if len(norm_t.split()) >= 2:
                    self.records.append({
                        "source": "kathbath", "idx": idx, "duration": dur, "text": norm_t, "audio_col": "audio_filepath"
                    })

        # Load IndicVoices train split
        if max_indicvoices > 0:
            print("[SFT Data] Indexing IndicVoices Gujarati train via metadata columns...")
            ds_iv = load_dataset("ai4bharat/IndicVoices", "gujarati", split="train", cache_dir=str(DATA_CACHE), token=token)
            ds_iv, _ = _cast_audio_columns_without_decoding(ds_iv)
            self.datasets["indicvoices"] = ds_iv

            iv_durs = ds_iv["duration"]
            iv_texts = ds_iv["text"]
            iv_candidates = [
                (idx, float(d), txt)
                for idx, (d, txt) in enumerate(zip(iv_durs, iv_texts))
                if d is not None and min_dur <= float(d) <= max_dur and txt and len(str(txt).split()) >= 3
            ]
            rng.shuffle(iv_candidates)
            for idx, dur, txt in iv_candidates[:max_indicvoices]:
                norm_t = normalize_gujarati(txt)
                if len(norm_t.split()) >= 3:
                    self.records.append({
                        "source": "indicvoices", "idx": idx, "duration": dur, "text": norm_t, "audio_col": "audio_filepath"
                    })

        total_hours = sum(r["duration"] for r in self.records) / 3600.0
        print(f"[SFT Data] Built balanced corpus: {len(self.records):,} clips ({total_hours:.2f} hours)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        ds = self.datasets[rec["source"]]
        row = ds[int(rec["idx"])]
        audio_raw, sr = _decode_audio_value(row[rec["audio_col"]])
        audio = _to_mono_16k(audio_raw, sr)
        # Cap audio length at 24 seconds to prevent OOM
        max_len = int(24.0 * TARGET_SR)
        if len(audio) > max_len:
            audio = audio[:max_len]
        return {
            "audio": audio,
            "text": rec["text"],
            "duration": len(audio) / TARGET_SR,
        }


# =====================================================================
# 2. BATCHING & COLLATION (GPU OPTIMIZATION)
# =====================================================================
class DurationBucketBatchSampler(Sampler[list[int]]):
    """groups training clips of similar audio duration into each batch to minimize padding waste."""

    def __init__(self, durations: Sequence[float], batch_size: int, bucket_multiplier: int = 25, seed: int = 42):
        self.durations = list(durations)
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_multiplier
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.durations)))
        rng.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: self.durations[idx])
            for b_start in range(0, len(bucket), self.batch_size):
                batches.append(bucket[b_start : b_start + self.batch_size])

        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.durations) / self.batch_size)


class Gemma4ASRCollator:
    """formats audio + prompt + target transcript and masks prompt/audio tokens with -100."""

    def __init__(self, processor: Any):
        self.processor = processor
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_convs = []
        prompt_convs = []

        for f in features:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": f["audio"], "sampling_rate": TARGET_SR},
                    {"type": "text", "text": PROMPT},
                ],
            }
            asst_msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": f["text"]}],
            }
            full_convs.append([user_msg, asst_msg])
            prompt_convs.append([user_msg])

        # Apply chat template for the full conversation (prompt + target)
        batch = self.processor.apply_chat_template(
            full_convs, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt", padding=True
        )
        # Apply chat template for just the prompt to find the exact split point
        prompt_batch = self.processor.apply_chat_template(
            prompt_convs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        prompt_attn = prompt_batch["attention_mask"]

        # Mask prompt and padding tokens so loss is ONLY calculated on the Gujarati transcript
        for i in range(len(features)):
            prompt_len = int(prompt_attn[i].sum().item())
            labels[i, :prompt_len] = -100
            labels[i, batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch


# =====================================================================
# 3. INFERENCE ADAPTER (LOADING THE SFT MODEL LATER)
# =====================================================================
class Gemma4SFTAdapter(Gemma4Step5Adapter):
    """step 6 sft adapter: loads gemma 4 e4b + trained sft lora weights + step 5 logit mask."""

    batch_mode = "multimodal_sft_batch"

    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.name = "gemma-4-E4B-sft"
        print(f"[Model] Loading {self.name} from {checkpoint_dir}...")
        token = hf_auth_token()
        self.processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it", token=token)
        base_model = AutoModelForMultimodalLM.from_pretrained(
            "google/gemma-4-E4B-it", torch_dtype=torch.bfloat16, device_map="auto", token=token,
        )
        self.model = PeftModel.from_pretrained(base_model, str(checkpoint_dir)).eval()

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            banned_tensor = build_banned_script_token_tensor(tokenizer, self.model, self.model.device)
            self.logits_processors = LogitsProcessorList([GujaratiScriptLogitsProcessor(banned_tensor)])
        else:
            self.logits_processors = LogitsProcessorList()


# =====================================================================
# 4. TRAINING LOOP & MONKEY PATCH
# =====================================================================
def train_sft(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = CuratedGujaratiSFTDataset(
        max_fleurs=args.max_fleurs,
        max_kathbath=args.max_kathbath,
        max_indicvoices=args.max_indicvoices,
    )
    durations = [r["duration"] for r in dataset.records]
    batch_sampler = DurationBucketBatchSampler(durations, batch_size=args.batch_size)

    token = hf_auth_token()
    print("[SFT Model] Loading google/gemma-4-E4B-it in bfloat16...")
    processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it", token=token)

    # -----------------------------------------------------------------
    # FIX: Monkey-patch Gemma4ClippableLinear so PEFT recognizes it.
    # We do this immediately before loading the model.
    # -----------------------------------------------------------------
    class PatchedClippableLinear(nn.Linear):
        def __init__(self, config, in_features, out_features):
            nn.Linear.__init__(self, in_features, out_features, bias=False)
            self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
            if self.use_clipped_linears:
                self.register_buffer("input_min", torch.tensor(-float("inf")))
                self.register_buffer("input_max", torch.tensor(float("inf")))
                self.register_buffer("output_min", torch.tensor(-float("inf")))
                self.register_buffer("output_max", torch.tensor(float("inf")))

        def forward(self, x):
            if self.use_clipped_linears:
                x = torch.clamp(x, self.input_min, self.input_max)
            out = nn.Linear.forward(self, x)
            if self.use_clipped_linears:
                out = torch.clamp(out, self.output_min, self.output_max)
            return out

    modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear
    # -----------------------------------------------------------------

    model = AutoModelForMultimodalLM.from_pretrained(
        "google/gemma-4-E4B-it", torch_dtype=torch.bfloat16, device_map="auto", token=token,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = Gemma4ASRCollator(processor)
    loader = DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=args.workers,
        collate_fn=collator, pin_memory=True, prefetch_factor=2 if args.workers > 0 else None,
    )

    total_micro_steps = len(loader) * args.epochs
    total_opt_steps = args.max_steps if args.max_steps > 0 else max(1, total_micro_steps // args.grad_accum)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01, fused=torch.cuda.is_available())
    warmup_steps = max(10, int(0.06 * total_opt_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    print(f"\n[SFT Start] micro_batch={args.batch_size}, grad_accum={args.grad_accum}, effective_batch={args.batch_size * args.grad_accum}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    opt_step = 0
    running_loss = 0.0
    total_clips = 0
    t_start = time.perf_counter()
    log_path = RESULTS_DIR / "step6_sft_train_log.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        batch_sampler.epoch = epoch
        for micro_step, batch in enumerate(loader, 1):
            batch_ = {k: (v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            outputs = model(**batch_)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            running_loss += float(outputs.loss.detach().item())
            total_clips += len(batch["input_ids"])

            if micro_step % args.grad_accum == 0 or micro_step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0 or opt_step == 1 or opt_step == total_opt_steps:
                    elapsed = time.perf_counter() - t_start
                    avg_loss = running_loss / max(1, args.log_every * args.grad_accum if opt_step > 1 else micro_step)
                    running_loss = 0.0
                    vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                    clips_per_s = total_clips / max(elapsed, 1e-6)
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"[SFT Step {opt_step:4d}/{total_opt_steps}] loss={avg_loss:.4f} | lr={lr_now:.2e} | speed={clips_per_s:.2f} clips/s | peak_vram={vram_gb:.1f} GB")
                    
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": opt_step, "loss": round(avg_loss, 4), "lr": lr_now, "clips_per_s": round(clips_per_s, 3), "peak_vram_gb": round(vram_gb, 2), "elapsed_s": round(elapsed, 1)}) + "\n")

                if args.save_every > 0 and opt_step % args.save_every == 0:
                    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(CHECKPOINT_DIR))
                    processor.save_pretrained(str(CHECKPOINT_DIR))
                    print(f"[SFT Checkpoint] Saved intermediate adapter to {CHECKPOINT_DIR} at step {opt_step}")

                if args.max_steps > 0 and opt_step >= args.max_steps:
                    break
        if args.max_steps > 0 and opt_step >= args.max_steps:
            break

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(CHECKPOINT_DIR))
    processor.save_pretrained(str(CHECKPOINT_DIR))
    total_min = (time.perf_counter() - t_start) / 60.0
    print(f"\n[SFT Done] Saved final LoRA checkpoint to {CHECKPOINT_DIR} in {total_min:.1f} minutes.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 6: High-Throughput Gemma 4 E4B Gujarati SFT")
    parser.add_argument("--max-fleurs", type=int, default=3145)
    parser.add_argument("--max-kathbath", type=int, default=8000)
    parser.add_argument("--max-indicvoices", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4, help="per-GPU microbatch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full epoch")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=150)
    args = parser.parse_args()
    train_sft(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
