#!/usr/bin/env python3
"""step 5: zero-training improvements for gemma 4 e4b gujarati asr."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import transformers
from tabulate import tabulate
from transformers import AutoModelForMultimodalLM, AutoProcessor, LogitsProcessor, LogitsProcessorList

from run_baselines_optimized import (
    DEFAULT_DB_PATH,
    EXPECTED_ROWS,
    RESULTS_DIR,
    TARGET_SR,
    BaseASRAdapter,
    _dataset_metrics,
    hf_auth_token,
)

# silence harmless processor_kwargs deprecation warnings during batched generation
transformers.logging.set_verbosity_error()

# match non-gujarati scripts and markdown symbols that cause script drift
FORBIDDEN_SCRIPT_RE = re.compile(
    r"["
    r"A-Za-z"                  # latin (english drift / chat preambles)
    r"\u0900-\u097F"           # devanagari (hindi / marathi drift)
    r"\u0980-\u0A7F"           # bengali, gurmukhi
    r"\u0B00-\u0DFF"           # oriya, tamil, telugu, kannada, malayalam
    r"\u0400-\u04FF"           # cyrillic
    r"\u0600-\u06FF"           # arabic
    r"\u4E00-\u9FFF"           # cjk
    r"*_`#<>~\[\]{}|\\"        # markdown / formatting clutter
    r"]"
)


class GujaratiScriptLogitsProcessor(LogitsProcessor):
    """sets logits of non-gujarati script tokens to -inf at every decoding step."""

    def __init__(self, banned_token_ids: torch.Tensor):
        self.banned_token_ids = banned_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.banned_token_ids.device != scores.device:
            self.banned_token_ids = self.banned_token_ids.to(scores.device)
        scores[:, self.banned_token_ids] = -float("inf")
        return scores


def build_banned_script_token_tensor(tokenizer, model, device: torch.device) -> torch.Tensor:
    # 1. collect all special, control, and stop token ids so generation stops normally
    safe_special_ids = set(range(256))
    safe_special_ids.update(getattr(tokenizer, "all_special_ids", []) or [])
    safe_special_ids.update(getattr(tokenizer, "added_tokens_decoder", {}).keys())

    for tok_name in ("<eos>", "<end_of_turn>", "<start_of_turn>", "<pad>", "<bos>", "<unk>"):
        tok_id = tokenizer.convert_tokens_to_ids(tok_name)
        if isinstance(tok_id, int) and tok_id >= 0:
            safe_special_ids.add(tok_id)

    for cfg in (
        tokenizer,
        getattr(model, "generation_config", None),
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        if cfg is None:
            continue
        for attr in ("eos_token_id", "pad_token_id", "bos_token_id", "unk_token_id"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int):
                safe_special_ids.add(val)
            elif isinstance(val, (list, tuple, set)):
                safe_special_ids.update(int(x) for x in val if x is not None)

    vocab_size = len(tokenizer)
    # skip_special_tokens=True ensures control tokens decode to "" and are never banned
    all_token_strs = tokenizer.batch_decode(
        [[i] for i in range(vocab_size)],
        skip_special_tokens=True,
    )

    banned_ids = [
        idx for idx, tok_str in enumerate(all_token_strs)
        if idx not in safe_special_ids
        and tok_str
        and not (tok_str.startswith("<") and tok_str.endswith(">"))
        and FORBIDDEN_SCRIPT_RE.search(tok_str)
    ]
    print(
        f"[Step 5 Logit Mask] Scanned {vocab_size:,} tokens: "
        f"banned {len(banned_ids):,} non-Gujarati tokens "
        f"(preserved {len(safe_special_ids):,} special/control tokens)."
    )
    return torch.tensor(banned_ids, dtype=torch.long, device=device)


class Gemma4Step5Adapter(BaseASRAdapter):
    """zero-training optimized gemma 4 e4b adapter."""

    batch_mode = "multimodal_step5_batch"
    PROMPT = (
        "Transcribe the following speech segment in Gujarati script. "
        "Output only the verbatim Gujarati transcription, with no explanation."
    )

    def __init__(self):
        self.name = "gemma-4-E4B-step5"
        print(f"[Model] Loading {self.name} (Gujarati Logit Mask + Loop Blocker + Silence Chunker)...")
        token = hf_auth_token()
        self.processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it", token=token)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            "google/gemma-4-E4B-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        ).eval()

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            banned_tensor = build_banned_script_token_tensor(tokenizer, self.model, self.model.device)
            self.logits_processors = LogitsProcessorList([GujaratiScriptLogitsProcessor(banned_tensor)])
        else:
            self.logits_processors = LogitsProcessorList()

    @classmethod
    def _split_audio_at_silence(cls, audio_array: np.ndarray) -> list[np.ndarray]:
        max_seconds = 28.0
        min_search_seconds = 20.0
        overlap_seconds = 0.40
        max_samples = int(max_seconds * TARGET_SR)

        if len(audio_array) <= max_samples:
            return [audio_array]

        win_len = int(0.10 * TARGET_SR)
        hop_len = int(0.02 * TARGET_SR)
        overlap_samples = int(overlap_seconds * TARGET_SR)

        chunks: list[np.ndarray] = []
        start = 0
        total_len = len(audio_array)

        while start < total_len:
            rem = total_len - start
            if rem <= max_samples:
                chunks.append(audio_array[start:])
                break

            search_lo = start + int(min_search_seconds * TARGET_SR)
            search_hi = start + max_samples - win_len
            best_cut = start + max_samples

            if search_hi > search_lo:
                min_energy = float("inf")
                for pos in range(search_lo, search_hi, hop_len):
                    energy = float(np.mean(np.abs(audio_array[pos : pos + win_len])))
                    if energy < min_energy:
                        min_energy = energy
                        best_cut = pos + (win_len // 2)

            chunks.append(audio_array[start:best_cut])
            start = max(start + 1, best_cut - overlap_samples)

        return chunks

    @staticmethod
    def _max_tokens_for_chunks(chunks: Sequence[np.ndarray]) -> int:
        max_dur_s = max((len(c) / TARGET_SR for c in chunks), default=10.0)
        return min(448, max(64, int(max_dur_s * 24)))

    def _messages(self, chunk: np.ndarray):
        return [{"role": "user", "content": [
            {"type": "audio", "audio": chunk, "sampling_rate": TARGET_SR},
            {"type": "text", "text": self.PROMPT},
        ]}]

    def _transcribe_chunk(self, chunk: np.ndarray) -> str:
        inputs = self.processor.apply_chat_template(
            self._messages(chunk),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        max_new = self._max_tokens_for_chunks([chunk])
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                no_repeat_ngram_size=5,
                logits_processor=self.logits_processors,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

    def _transcribe_chunks_batch(self, chunks: Sequence[np.ndarray]) -> list[str]:
        conversations = [self._messages(chunk) for chunk in chunks]
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        max_new = self._max_tokens_for_chunks(chunks)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                no_repeat_ngram_size=5,
                logits_processor=self.logits_processors,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        else:
            decoded = self.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [str(text).strip() for text in decoded]

    @staticmethod
    def _join_overlapping_transcripts(parts: list[str]) -> str:
        merged: list[str] = []
        for part in parts:
            words = part.split()
            if not words:
                continue
            overlap = 0
            max_overlap = min(10, len(merged), len(words))
            for n in range(max_overlap, 0, -1):
                if merged[-n:] == words[:n]:
                    overlap = n
                    break
            merged.extend(words[overlap:])
        return " ".join(merged).strip()

    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        chunks = self._split_audio_at_silence(audio_array)
        return self._join_overlapping_transcripts([self._transcribe_chunk(c) for c in chunks])

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

        chunk_batch_size = max(1, min(batch_size, 16))
        for start in range(0, len(chunks), chunk_batch_size):
            chunk_group = chunks[start : start + chunk_batch_size]
            ref_group = refs[start : start + chunk_batch_size]
            try:
                texts = self._transcribe_chunks_batch(chunk_group)
                for (sample_i, chunk_i), text in zip(ref_group, texts):
                    per_sample[sample_i][chunk_i] = text
            except Exception as batch_exc:
                print(f"\n[WARN] Gemma Step5 batch failed; retrying individually: {batch_exc!r}")
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
                outputs.append(RuntimeError("Gemma Step5 did not produce every audio chunk"))
            else:
                outputs.append(self._join_overlapping_transcripts([str(p) for p in parts]))
        return outputs


def compare_and_plot_step5(db_path: Path = DEFAULT_DB_PATH) -> None:
    """print delta table and save a side-by-side comparison plot for step 5."""
    conn = sqlite3.connect(str(db_path))
    metrics = _dataset_metrics(conn)
    conn.close()

    focus_models = ["gemma-4-E4B-it", "gemma-4-E4B-step5", "indic-transcribe-core"]
    datasets = ["fleurs", "indicvoices", "kathbath"]

    print("\n" + "=" * 110)
    print("STEP 5 ZERO-TRAINING IMPACT: BASELINE GEMMA vs. STEP-5 GEMMA vs. TARGET (BODHAN)")
    print("=" * 110)

    rows = []
    for ds in datasets:
        for m_name in focus_models:
            if (ds, m_name) not in metrics:
                continue
            m = metrics[(ds, m_name)]
            base = metrics.get((ds, "gemma-4-E4B-it"))
            cer_delta = ""
            wer_delta = ""
            if base and m_name == "gemma-4-E4B-step5":
                cer_delta = f" ({m['corpus_cer'] - base['corpus_cer']:+.2f}%)"
                wer_delta = f" ({m['corpus_wer'] - base['corpus_wer']:+.2f}%)"
            rows.append([
                ds,
                m_name,
                f"{m['n']}/{EXPECTED_ROWS.get(ds, '?')}",
                f"{m['corpus_cer']:.2f}%{cer_delta}",
                f"{m['corpus_wer']:.2f}%{wer_delta}",
                f"{m['mean_capped_cer']:.2f}%",
                f"{m['drift_pct']:.2f}%",
                f"{m['loop_pct']:.2f}%",
                f"{m['median_rtf']:.4f}",
            ])

    print(tabulate(
        rows,
        headers=["Dataset", "Model", "N", "Corpus CER", "Corpus WER", "Capped CER", "Drift %", "Loop %", "Median RTF"],
        tablefmt="github",
    ))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        charts_dir = RESULTS_DIR / "charts_optimized"
        charts_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metric_specs = [
            ("corpus_cer", "Corpus CER (%) — Lower is Better"),
            ("corpus_wer", "Corpus WER (%) — Lower is Better"),
            ("drift_pct", "Script Drift (%) — Lower is Better"),
        ]
        colors = {
            "gemma-4-E4B-it": "#ef4444",
            "gemma-4-E4B-step5": "#3b82f6",
            "indic-transcribe-core": "#10b981",
        }

        x = np.arange(len(datasets))
        width = 0.26

        for ax, (m_key, title) in zip(axes, metric_specs):
            for idx, m_name in enumerate(focus_models):
                vals = [metrics.get((ds, m_name), {}).get(m_key, 0.0) for ds in datasets]
                bars = ax.bar(x + (idx - 1) * width, vals, width, label=m_name, color=colors[m_name])
                for bar, val in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            ax.set_title(title)
            ax.set_xticks(x, datasets)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8)

        fig.suptitle("Step 5 Zero-Training Improvements: Baseline Gemma vs. Step 5 Gemma vs. Target")
        fig.tight_layout()
        out_plot = charts_dir / "step5_gemma_comparison.png"
        fig.savefig(out_plot, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[Step 5 Plot] Saved comparison chart to: {out_plot}")
    except Exception as exc:
        print(f"[WARN] Could not save Step 5 comparison chart: {exc!r}")


if __name__ == "__main__":
    compare_and_plot_step5()
