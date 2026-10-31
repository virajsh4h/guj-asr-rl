#!/usr/bin/env python3
"""Gujarati ASR baseline runner.

- FLEURS Gujarati test, IndicVoices Gujarati valid, Kathbath Gujarati valid
- Audio is read with Audio(decode=False), so datasets does not require torchcodec
- Recording-index sample IDs avoid FLEURS prompt-ID collisions
- Inference failures are logged separately and are not stored as empty predictions
- Per-dataset metrics are corpus WER/CER; macro scores require complete dataset coverage
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sqlite3
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from huggingface_hub import get_token, hf_hub_download, snapshot_download
from jiwer import cer as corpus_cer
from jiwer import wer as corpus_wer
from scipy.signal import resample_poly
from tabulate import tabulate
from tqdm import tqdm

from gu_normalizer import compute_utterance_metrics


WORKSPACE = Path(__file__).resolve().parent
DATA_CACHE = WORKSPACE / "data_cache"
MODEL_CACHE = WORKSPACE / "model_cache"
RESULTS_DIR = WORKSPACE / "results"
DB_PATH = RESULTS_DIR / "predictions_v2.sqlite"
FAILURE_PATH = RESULTS_DIR / "inference_failures.jsonl"
TARGET_SR = 16_000

DATASET_SPLITS = {
    "fleurs": "test",
    "indicvoices": "valid",
    "kathbath": "valid",
}

MODEL_KEYS = ["whisper", "whisper-gu", "indicconformer", "sravaani", "bodhan", "gemma4"]


def hf_auth_token() -> str | bool | None:
    """Use HF_TOKEN when explicitly set; otherwise use `hf auth login` credentials."""
    return os.environ.get("HF_TOKEN") or get_token() or None


def init_db() -> sqlite3.Connection:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            dataset TEXT NOT NULL,
            split TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            audio_duration_s REAL,
            reference_raw TEXT,
            hypothesis_raw TEXT,
            reference_norm TEXT,
            hypothesis_norm TEXT,
            wer REAL,
            cer REAL,
            capped_cer REAL,
            latency_s REAL,
            rtf REAL,
            script_drift INTEGER,
            drift_ratio REAL,
            loop_flag INTEGER,
            failure_tag TEXT DEFAULT '',
            PRIMARY KEY (dataset, split, sample_id, model_name)
        )
        """
    )
    conn.commit()
    return conn


def is_evaluated(conn: sqlite3.Connection, item: dict[str, Any], model_name: str) -> bool:
    row = conn.execute(
        """SELECT 1 FROM predictions
           WHERE dataset=? AND split=? AND sample_id=? AND model_name=?""",
        (item["dataset"], item["split"], item["sample_id"], model_name),
    ).fetchone()
    return row is not None


def _cast_audio_columns_without_decoding(ds):
    """Disable HF Audio decoding for every Audio feature (including audio_filepath)."""
    audio_columns = [
        name for name, feature in ds.features.items()
        if isinstance(feature, Audio)
    ]
    if not audio_columns:
        # Some converted parquet datasets may expose a path-like column without
        # retaining the Audio feature metadata. Keep these as raw values.
        audio_columns = [
            name for name in ("audio", "audio_filepath")
            if name in ds.column_names
        ]
    if not audio_columns:
        raise RuntimeError(
            f"No Audio feature found. Columns={ds.column_names}; features={ds.features}"
        )

    for column in audio_columns:
        try:
            ds = ds.cast_column(column, Audio(decode=False))
        except Exception as exc:
            raise RuntimeError(
                f"Could not disable automatic audio decoding for column '{column}': {exc}"
            ) from exc
    return ds, audio_columns


def _decode_audio_value(value: Any) -> tuple[np.ndarray, int]:
    """Decode an Audio(decode=False) dict, file path, or already-decoded array."""
    if isinstance(value, dict):
        if value.get("array") is not None:
            arr = np.asarray(value["array"], dtype=np.float32)
            sr = int(value.get("sampling_rate") or TARGET_SR)
            return arr, sr
        blob = value.get("bytes")
        path = value.get("path")
        if blob:
            try:
                arr, sr = sf.read(io.BytesIO(blob), dtype="float32")
                return np.asarray(arr, dtype=np.float32), int(sr)
            except Exception as soundfile_error:
                # Fallback for encodings unsupported by the installed libsndfile.
                if shutil.which("ffmpeg"):
                    proc = subprocess.run(
                        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                         "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "pipe:1"],
                        input=blob,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    if proc.returncode == 0 and proc.stdout:
                        return np.frombuffer(proc.stdout, dtype="<f4").copy(), TARGET_SR
                raise RuntimeError(
                    f"Could not decode embedded audio bytes with soundfile: {soundfile_error}"
                ) from soundfile_error
        if path:
            path = str(path)
            if os.path.isfile(path):
                try:
                    arr, sr = sf.read(path, dtype="float32")
                    return np.asarray(arr, dtype=np.float32), int(sr)
                except Exception as soundfile_error:
                    if shutil.which("ffmpeg"):
                        proc = subprocess.run(
                            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
                             "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "pipe:1"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=False,
                        )
                        if proc.returncode == 0 and proc.stdout:
                            return np.frombuffer(proc.stdout, dtype="<f4").copy(), TARGET_SR
                    raise RuntimeError(
                        f"Could not decode audio path '{path}': {soundfile_error}"
                    ) from soundfile_error
        raise RuntimeError(f"Audio dict has neither usable bytes, path, nor array: {list(value)}")

    if isinstance(value, (str, os.PathLike)):
        path = str(value)
        arr, sr = sf.read(path, dtype="float32")
        return np.asarray(arr, dtype=np.float32), int(sr)

    raise TypeError(f"Unsupported audio value type: {type(value).__name__}")


def _to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        # soundfile returns [samples, channels]. Also tolerate [channels, samples].
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    elif arr.ndim != 1:
        raise ValueError(f"Expected 1D/2D audio, got shape={arr.shape}")

    arr = np.nan_to_num(arr, copy=False)
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")
    if sample_rate != TARGET_SR:
        from math import gcd
        g = gcd(int(sample_rate), TARGET_SR)
        arr = resample_poly(arr, TARGET_SR // g, int(sample_rate) // g).astype(np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_gujarati_benchmark(dataset_key: str, max_samples: int = 0) -> Iterator[dict[str, Any]]:
    """Yield one audio example at a time; never eagerly materialize the audio split."""
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    token = hf_auth_token()

    if dataset_key == "fleurs":
        print("[Dataset] Loading pinned FLEURS Gujarati test Parquet...")
        parquet_path = hf_hub_download(
            repo_id="google/fleurs",
            repo_type="dataset",
            filename="gu_in/test-00000-of-00001.parquet",
            revision="fdd74f7cdf01fc7641ab3072b6a38d364b3667f6",
            cache_dir=str(DATA_CACHE),
            token=token,
        )
        ds = load_dataset(
            "parquet",
            data_files={"test": parquet_path},
            split="test",
            cache_dir=str(DATA_CACHE),
        )
        split_name = "test"
        text_candidates = ["transcription", "raw_transcription", "text"]

    elif dataset_key == "indicvoices":
        print("[Dataset] Loading ai4bharat/IndicVoices Gujarati valid...")
        ds = load_dataset(
            "ai4bharat/IndicVoices",
            "gujarati",
            split="valid",
            cache_dir=str(DATA_CACHE),
            token=token,
        )
        split_name = "valid"
        text_candidates = ["text", "normalized", "verbatim", "transcription"]

    elif dataset_key == "kathbath":
        print("[Dataset] Loading ai4bharat/Kathbath Gujarati valid...")
        ds = load_dataset(
            "ai4bharat/Kathbath",
            "gujarati",
            split="valid",
            cache_dir=str(DATA_CACHE),
            token=token,
        )
        split_name = "valid"
        text_candidates = ["text", "transcription"]

    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    ds, audio_columns = _cast_audio_columns_without_decoding(ds)
    audio_column = next((c for c in ("audio", "audio_filepath") if c in audio_columns), audio_columns[0])
    text_column = next((c for c in text_candidates if c in ds.column_names), None)
    if text_column is None:
        raise RuntimeError(
            f"Could not identify transcript column. Candidates={text_candidates}; columns={ds.column_names}"
        )

    n_total = len(ds)
    n_limit = n_total if max_samples <= 0 else min(n_total, max_samples)
    print(f"[Dataset] {dataset_key}: split={split_name}, rows={n_total}, evaluating={n_limit}, audio={audio_column}, text={text_column}")

    for idx in range(n_limit):
        row = ds[idx]
        # Use row position. FLEURS' `id` identifies the sentence prompt and is
        # repeated across speakers, so it is NOT a unique recording ID.
        sample_id = f"{dataset_key}_{idx:06d}"
        audio_raw, sr = _decode_audio_value(row[audio_column])
        audio = _to_mono_16k(audio_raw, sr)
        duration_s = len(audio) / TARGET_SR
        reference = str(row.get(text_column, "") or "").strip()

        if not reference or duration_s < 0.30:
            continue

        yield {
            "dataset": dataset_key,
            "split": split_name,
            "row_index": idx,
            "sample_id": sample_id,
            "audio_array": audio,
            "duration_s": round(duration_s, 4),
            "reference": reference,
        }


class BaseASRAdapter:
    name = "base"

    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        raise NotImplementedError


class WhisperAdapter(BaseASRAdapter):
    def __init__(self, model_id: str, name: str, gujarati_finetune: bool = False):
        from transformers import GenerationConfig, pipeline

        self.name = name
        self.device = 0 if torch.cuda.is_available() else -1
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"[Model] Loading {name} ({model_id})...")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=self.device,
            torch_dtype=dtype,
            chunk_length_s=30,
            token=hf_auth_token(),
        )

        # Explicitly force Gujarati transcription. For the older Gujarati
        # fine-tune, replace its tiny legacy generation config with Whisper's
        # base config before setting the explicit decoder prefix.
        if gujarati_finetune:
            self.pipe.model.generation_config = GenerationConfig.from_pretrained(
                "openai/whisper-small",
                token=hf_auth_token(),
            )
        forced_ids = self.pipe.tokenizer.get_decoder_prompt_ids(
            language="gu", task="transcribe"
        )
        self.pipe.model.config.forced_decoder_ids = forced_ids
        self.pipe.model.generation_config.forced_decoder_ids = forced_ids

    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        result = self.pipe(
            {"array": audio_array, "sampling_rate": TARGET_SR},
            generate_kwargs={"max_new_tokens": 224},
        )
        return str(result.get("text", "")).strip()


class Gemma4Adapter(BaseASRAdapter):
    def __init__(self):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.name = "gemma-4-E4B-it"
        print(f"[Model] Loading {self.name}...")
        token = hf_auth_token()
        self.processor = AutoProcessor.from_pretrained(
            "google/gemma-4-E4B-it", token=token
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            "google/gemma-4-E4B-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        ).eval()

    def _transcribe_chunk(self, chunk: np.ndarray) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": chunk, "sampling_rate": TARGET_SR},
                {"type": "text", "text": (
                    "Transcribe the following speech segment in Gujarati script. "
                    "Output only the verbatim Gujarati transcription, with no explanation."
                )},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=224, do_sample=False
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()

    @staticmethod
    def _join_overlapping_transcripts(parts: list[str]) -> str:
        """Remove exact word overlap caused by the 0.75-second audio overlap."""
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
        # Gemma's audio frontend is designed for short segments. Chunk long
        # recordings explicitly rather than silently truncating or failing.
        max_seconds = 28.0
        overlap_seconds = 0.75
        max_samples = int(max_seconds * TARGET_SR)
        step = int((max_seconds - overlap_seconds) * TARGET_SR)
        if len(audio_array) <= max_samples:
            return self._transcribe_chunk(audio_array)

        parts = []
        start = 0
        while start < len(audio_array):
            chunk = audio_array[start:start + max_samples]
            if len(chunk) == 0:
                break
            parts.append(self._transcribe_chunk(chunk))
            if start + max_samples >= len(audio_array):
                break
            start += step
        return self._join_overlapping_transcripts(parts)


class IndicConformerAdapter(BaseASRAdapter):
    """Load the official multilingual HF/ONNX wrapper without a NeMo fallback."""
    def __init__(self):
        from transformers import AutoModel

        self.name = "indicconformer-600m"
        local_dir = MODEL_CACHE / "indic-conformer-600m-multilingual"
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        print(f"[Model] Downloading/materializing IndicConformer in {local_dir}...")
        model_dir = snapshot_download(
            repo_id="ai4bharat/indic-conformer-600m-multilingual",
            local_dir=str(local_dir),
            token=hf_auth_token(),
        )
        print("[Model] Loading IndicConformer HF wrapper from local files...")
        self.model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        # This wrapper uses ONNX Runtime; it is not a regular torch.nn.Module.
        # Leave its waveform tensor on CPU as in AI4Bharat's example.

        try:
            import onnxruntime as ort
            print(f"[Model] IndicConformer ONNX providers: {ort.get_available_providers()}")
        except Exception:
            pass

    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        waveform = torch.from_numpy(np.asarray(audio_array, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            result = self.model(waveform, "gu", "rnnt")
        if isinstance(result, (list, tuple)):
            result = result[0]
        return str(result).strip()


class SraVaaniAdapter(BaseASRAdapter):
    def __init__(self):
        from transformers import AutoModel

        self.name = "sravaani-1.0"
        print("[Model] Loading SraVaani through official Transformers adapter...")
        self.model = AutoModel.from_pretrained(
            "ARTPARK-IISc/SraVaani-1.0",
            trust_remote_code=True,
            token=hf_auth_token(),
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        print("[Model] SraVaani loaded successfully.")

    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        fd, tmp_path = tempfile.mkstemp(prefix="sravaani_", suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp_path, audio_array, TARGET_SR, subtype="PCM_16")
            with torch.inference_mode():
                hypotheses = self.model.transcribe(tmp_path, return_hypotheses=True)
            if not hypotheses:
                return ""
            first = hypotheses[0] if isinstance(hypotheses, (list, tuple)) else hypotheses
            return str(getattr(first, "text", first)).strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class BodhanAdapter(BaseASRAdapter):
    def __init__(self):
        self.name = "indic-transcribe-core"
        local_dir = MODEL_CACHE / "indic-transcribe-core"
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        print("[Model] Downloading BodhanAI Indic-Transcribe-core...")
        model_dir = snapshot_download(
            repo_id="bodhan-ai/indic-transcribe-core",
            local_dir=str(local_dir),
            token=hf_auth_token(),
        )
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from indic_transcribe import IndicTranscribe
        self.asr = IndicTranscribe.from_pretrained(model_dir)

    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        fd, tmp_path = tempfile.mkstemp(prefix="bodhan_", suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp_path, audio_array, TARGET_SR, subtype="PCM_16")
            result = self.asr(tmp_path, lang="gu")
            return str(result).strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def build_adapter(key: str) -> BaseASRAdapter:
    if key == "whisper":
        return WhisperAdapter("openai/whisper-small", "whisper-small")
    if key == "whisper-gu":
        return WhisperAdapter(
            "vasista22/whisper-gujarati-small",
            "whisper-gujarati-small",
            gujarati_finetune=True,
        )
    if key == "gemma4":
        return Gemma4Adapter()
    if key == "indicconformer":
        return IndicConformerAdapter()
    if key == "sravaani":
        return SraVaaniAdapter()
    if key == "bodhan":
        return BodhanAdapter()
    raise ValueError(f"Unknown model key: {key}")


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def log_inference_failure(item: dict[str, Any], model_name: str, error: Exception) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": item["dataset"],
        "split": item["split"],
        "sample_id": item["sample_id"],
        "row_index": item["row_index"],
        "model": model_name,
        "error": repr(error),
    }
    with FAILURE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _dataset_metrics(conn: sqlite3.Connection) -> dict[tuple[str, str], dict[str, Any]]:
    rows = conn.execute(
        """SELECT dataset, split, model_name, reference_norm, hypothesis_norm,
                  capped_cer, rtf, script_drift, loop_flag
           FROM predictions
           ORDER BY dataset, model_name, row_index"""
    ).fetchall()
    grouped: dict[tuple[str, str], list[tuple]] = {}
    for row in rows:
        grouped.setdefault((row[0], row[2]), []).append(row)

    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for (dataset, model_name), items in grouped.items():
        pairs = [(r[3] or "", r[4] or "") for r in items if r[3]]
        refs = [p[0] for p in pairs]
        hyps = [p[1] for p in pairs]
        metrics[(dataset, model_name)] = {
            "split": items[0][1],
            "n": len(items),
            "corpus_wer": 100.0 * corpus_wer(refs, hyps) if refs else 0.0,
            "corpus_cer": 100.0 * corpus_cer(refs, hyps) if refs else 0.0,
            "mean_capped_cer": float(np.mean([r[5] for r in items])) if items else 0.0,
            "median_rtf": float(median([r[6] for r in items])) if items else 0.0,
            "drift_pct": 100.0 * sum(int(r[7] or 0) for r in items) / len(items) if items else 0.0,
            "loop_pct": 100.0 * sum(int(r[8] or 0) for r in items) / len(items) if items else 0.0,
            "refs": refs,
            "hyps": hyps,
        }
    return metrics


def print_leaderboard(conn: sqlite3.Connection, expected_datasets: list[str]) -> None:
    metrics = _dataset_metrics(conn)
    if not metrics:
        print("No predictions logged yet.")
        return

    print("\n" + "=" * 112)
    print("1. PER-DATASET GUJARATI ASR LEADERBOARD (CORPUS WER/CER)")
    print("=" * 112)
    table = []
    for (dataset, model), m in sorted(metrics.items()):
        table.append([
            f"{dataset} ({m['split']})", model, m["n"],
            f"{m['corpus_wer']:.2f}%", f"{m['corpus_cer']:.2f}%",
            f"{m['mean_capped_cer']:.2f}%", f"{m['drift_pct']:.1f}%",
            f"{m['loop_pct']:.1f}%", f"{m['median_rtf']:.4f}",
        ])
    print(tabulate(table, headers=[
        "Dataset", "Model", "N", "Corpus WER", "Corpus CER",
        "Mean Capped CER", "Drift %", "Loop %", "Median RTF",
    ], tablefmt="github"))

    models = sorted({model for _, model in metrics})
    print("\n" + "=" * 112)
    print("2. COVERAGE (MISSING DATA IS NOT COUNTED AS ZERO)")
    print("=" * 112)
    coverage_table = []
    for model in models:
        present = [d for d in expected_datasets if (d, model) in metrics]
        missing = [d for d in expected_datasets if (d, model) not in metrics]
        coverage_table.append([
            model,
            f"{len(present)}/{len(expected_datasets)}",
            ", ".join(present) or "—",
            ", ".join(missing) or "—",
        ])
    print(tabulate(coverage_table, headers=["Model", "Datasets", "Present", "Missing"], tablefmt="github"))

    print("\n" + "=" * 112)
    print("3. THREE-DATASET MACRO / POOLED LEADERBOARD (COMPLETE COVERAGE ONLY)")
    print("=" * 112)
    macro_table = []
    for model in models:
        if not all((d, model) in metrics for d in expected_datasets):
            continue
        per_dataset = [metrics[(d, model)] for d in expected_datasets]
        macro_wer = float(np.mean([m["corpus_wer"] for m in per_dataset]))
        macro_cer = float(np.mean([m["corpus_cer"] for m in per_dataset]))
        all_refs = [ref for m in per_dataset for ref in m["refs"]]
        all_hyps = [hyp for m in per_dataset for hyp in m["hyps"]]
        pooled_wer = 100.0 * corpus_wer(all_refs, all_hyps) if all_refs else 0.0
        pooled_cer = 100.0 * corpus_cer(all_refs, all_hyps) if all_refs else 0.0
        macro_table.append([
            model,
            sum(m["n"] for m in per_dataset),
            f"{macro_wer:.2f}%", f"{macro_cer:.2f}%",
            f"{pooled_wer:.2f}%", f"{pooled_cer:.2f}%",
        ])

    if macro_table:
        macro_table.sort(key=lambda r: float(r[3].rstrip("%")))
        print(tabulate(macro_table, headers=[
            "Model", "Total N", "Macro WER", "Macro CER", "Pooled WER", "Pooled CER"
        ], tablefmt="github"))
    else:
        print("No model has complete coverage across all selected datasets yet.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gujarati ASR baseline evaluator")
    parser.add_argument(
        "--datasets", nargs="+", default=["fleurs", "indicvoices", "kathbath"],
        choices=list(DATASET_SPLITS),
    )
    parser.add_argument(
        "--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS,
    )
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 evaluates the full split; positive values limit rows for smoke tests")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.max_samples < 0:
        parser.error("--max-samples must be >= 0")

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = init_db()

    if args.summary_only:
        print_leaderboard(conn, args.datasets)
        conn.close()
        return 0

    init_failures: list[tuple[str, str]] = []
    try:
        for model_key in args.models:
            try:
                adapter = build_adapter(model_key)
            except Exception as exc:
                print(f"[ERROR] Could not initialize model '{model_key}': {exc!r}")
                init_failures.append((model_key, repr(exc)))
                continue

            for dataset_key in args.datasets:
                print(f"\n[Run] model={adapter.name} dataset={dataset_key}")
                samples = load_gujarati_benchmark(dataset_key, args.max_samples)
                processed = 0
                failed = 0
                skipped = 0

                # Dataset-loader errors deliberately propagate: silently accepting
                # a partially yielded dataset would make the leaderboard misleading.
                for item in tqdm(samples, desc=f"{adapter.name} on {dataset_key}"):
                    if is_evaluated(conn, item, adapter.name):
                        skipped += 1
                        continue

                    try:
                        _synchronize_cuda()
                        start = time.perf_counter()
                        hypothesis = adapter.transcribe(item["audio_array"], item["row_index"])
                        _synchronize_cuda()
                        latency = time.perf_counter() - start
                    except Exception as exc:
                        failed += 1
                        log_inference_failure(item, adapter.name, exc)
                        print(f"\n[WARN] Inference failed: {item['sample_id']}: {exc!r}")
                        continue

                    metrics = compute_utterance_metrics(item["reference"], hypothesis)
                    rtf = latency / max(item["duration_s"], 0.1)
                    conn.execute(
                        """INSERT OR REPLACE INTO predictions
                           (dataset, split, row_index, sample_id, model_name,
                            audio_duration_s, reference_raw, hypothesis_raw,
                            reference_norm, hypothesis_norm, wer, cer, capped_cer,
                            latency_s, rtf, script_drift, drift_ratio, loop_flag, failure_tag)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            item["dataset"], item["split"], item["row_index"], item["sample_id"],
                            adapter.name, item["duration_s"], item["reference"], hypothesis,
                            metrics["ref_norm"], metrics["hyp_norm"], metrics["wer"], metrics["cer"],
                            metrics["capped_cer"], round(latency, 4), round(rtf, 4),
                            metrics["script_drift"], metrics["drift_ratio"], metrics["loop_flag"], "",
                        ),
                    )
                    conn.commit()
                    processed += 1

                print(
                    f"[Done] {adapter.name} / {dataset_key}: "
                    f"new={processed}, skipped_existing={skipped}, inference_failures={failed}"
                )

            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print_leaderboard(conn, args.datasets)
        if init_failures:
            print("\n[Model initialization failures]")
            for model, error in init_failures:
                print(f"- {model}: {error}")
        return 0 if not init_failures else 2

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
