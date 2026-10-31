#!/usr/bin/env python3
"""Optimized Gujarati ASR benchmark runner.

- FLEURS Gujarati test, IndicVoices Gujarati valid, Kathbath Gujarati valid
- Audio is read with Audio(decode=False), so datasets does not require torchcodec
- Recording-index sample IDs avoid FLEURS prompt-ID collisions
- Inference failures are logged separately and are not stored as empty predictions
- Per-dataset metrics are corpus WER/CER; macro scores require complete dataset coverage
- Direct SDPA batched Whisper inference with max_new_tokens=440 (fixes Gujarati BPE truncation)
"""

from __future__ import annotations

import argparse
import csv
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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
from typing import Any, Iterator, Sequence

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
DEFAULT_DB_PATH = RESULTS_DIR / "predictions_optimized.sqlite"
DEFAULT_FAILURE_PATH = RESULTS_DIR / "inference_failures_optimized.jsonl"
PERFORMANCE_PATH = RESULTS_DIR / "performance_optimized.jsonl"
SHM_DIR = Path("/dev/shm") if Path("/dev/shm").exists() else Path(tempfile.gettempdir())
TARGET_SR = 16_000

DATASET_SPLITS = {
    "fleurs": "test",
    "indicvoices": "valid",
    "kathbath": "valid",
}

MODEL_KEYS = ["whisper", "whisper-gu", "indicconformer", "sravaani", "bodhan", "gemma4"]
DEFAULT_MODEL_KEYS = ["bodhan", "sravaani", "gemma4", "whisper-gu", "whisper"]
MODEL_NAME_BY_KEY = {
    "whisper": "whisper-small",
    "whisper-gu": "whisper-gujarati-small",
    "gemma4": "gemma-4-E4B-it",
    "indicconformer": "indicconformer-600m",
    "sravaani": "sravaani-1.0",
    "bodhan": "indic-transcribe-core",
}
# IndicVoices valid has 2,970 raw rows; 2 rows are filtered (<0.30s or empty text), leaving 2,968 valid rows.
EXPECTED_ROWS = {"fleurs": 1000, "indicvoices": 2968, "kathbath": 2905}


def hf_auth_token() -> str | bool | None:
    """Use HF_TOKEN when explicitly set; otherwise use `hf auth login` credentials."""
    return os.environ.get("HF_TOKEN") or get_token() or None


def init_db(db_path: Path) -> sqlite3.Connection:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60)
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


def load_gujarati_dataset(dataset_key: str):
    """Load one pinned benchmark split without decoding audio eagerly."""
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    token = hf_auth_token()

    if dataset_key == "fleurs":
        print("[Dataset] Loading pinned FLEURS Gujarati test Parquet...")
        parquet_path = hf_hub_download(
            repo_id="google/fleurs", repo_type="dataset",
            filename="gu_in/test-00000-of-00001.parquet",
            revision="fdd74f7cdf01fc7641ab3072b6a38d364b3667f6",
            cache_dir=str(DATA_CACHE), token=token,
        )
        ds = load_dataset("parquet", data_files={"test": parquet_path},
                          split="test", cache_dir=str(DATA_CACHE))
        split_name = "test"
        text_candidates = ["transcription", "raw_transcription", "text"]
    elif dataset_key == "indicvoices":
        print("[Dataset] Loading ai4bharat/IndicVoices Gujarati valid...")
        ds = load_dataset("ai4bharat/IndicVoices", "gujarati", split="valid",
                          cache_dir=str(DATA_CACHE), token=token)
        split_name = "valid"
        text_candidates = ["text", "normalized", "verbatim", "transcription"]
    elif dataset_key == "kathbath":
        print("[Dataset] Loading ai4bharat/Kathbath Gujarati valid...")
        ds = load_dataset("ai4bharat/Kathbath", "gujarati", split="valid",
                          cache_dir=str(DATA_CACHE), token=token)
        split_name = "valid"
        text_candidates = ["text", "transcription"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    ds, audio_columns = _cast_audio_columns_without_decoding(ds)
    audio_column = next((c for c in ("audio", "audio_filepath") if c in audio_columns), audio_columns[0])
    text_column = next((c for c in text_candidates if c in ds.column_names), None)
    if text_column is None:
        raise RuntimeError(f"Could not identify transcript column. Candidates={text_candidates}; columns={ds.column_names}")
    print(f"[Dataset] {dataset_key}: split={split_name}, rows={len(ds)}, audio={audio_column}, text={text_column}")
    return ds, split_name, audio_column, text_column


def _decode_dataset_row(ds, dataset_key: str, split_name: str,
                        audio_column: str, text_column: str, idx: int):
    row = ds[idx]
    audio_raw, sr = _decode_audio_value(row[audio_column])
    audio = _to_mono_16k(audio_raw, sr)
    duration_s = len(audio) / TARGET_SR
    reference = str(row.get(text_column, "") or "").strip()
    if not reference or duration_s < 0.30:
        return None
    return {
        "dataset": dataset_key,
        "split": split_name,
        "row_index": idx,
        "sample_id": f"{dataset_key}_{idx:06d}",
        "audio_array": audio,
        "duration_s": round(duration_s, 4),
        "reference": reference,
    }


def _bounded_parallel_map(fn, iterable, workers: int, max_pending: int):
    """Ordered thread prefetch with a bounded number of in-flight audio decodes."""
    iterator = iter(iterable)
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="audio-decode") as pool:
        pending = deque()
        for _ in range(max(1, max_pending)):
            try:
                value = next(iterator)
            except StopIteration:
                break
            pending.append(pool.submit(fn, value))
        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                value = next(iterator)
            except StopIteration:
                continue
            pending.append(pool.submit(fn, value))


def iter_preprocessed_items(ds, dataset_key: str, split_name: str,
                            audio_column: str, text_column: str,
                            max_samples: int, already_done: set[str],
                            workers: int, prefetch: int):
    n_limit = len(ds) if max_samples <= 0 else min(len(ds), max_samples)
    indices = (i for i in range(n_limit) if f"{dataset_key}_{i:06d}" not in already_done)
    def decode_one(i):
        return _decode_dataset_row(ds, dataset_key, split_name,
                                   audio_column, text_column, i)
    for item in _bounded_parallel_map(decode_one, indices, workers, max(prefetch, workers)):
        if item is not None:
            yield item


class BaseASRAdapter:
    name = "base"
    batch_mode = "serial_fallback"

    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        raise NotImplementedError

    def transcribe_batch(self, audio_arrays: Sequence[np.ndarray],
                         row_indices: Sequence[int], batch_size: int) -> list[str | Exception]:
        outputs: list[str | Exception] = []
        for audio, row_index in zip(audio_arrays, row_indices):
            try:
                outputs.append(self.transcribe(audio, row_index))
            except Exception as exc:
                outputs.append(exc)
        return outputs


class WhisperAdapter(BaseASRAdapter):
    """Direct AutoProcessor + AutoModelForSpeechSeq2Seq SDPA batched adapter.

    Bypasses transformers.pipeline chunk-iterator overhead and raises max_new_tokens
    to 440 so Gujarati byte-level BPE sequences are not truncated mid-sentence.
    """
    batch_mode = "seq2seq_sdpa_batch"
    MAX_NEW_TOKENS = 440

    def __init__(self, model_id: str, name: str, gujarati_finetune: bool = False):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig

        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"[Model] Loading {name} ({model_id}) with direct SDPA batched backend...")

        token = hf_auth_token()
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        except Exception:
            self.processor = AutoProcessor.from_pretrained("openai/whisper-small", token=token)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            token=token,
        ).to(self.device).eval()

        # Always normalize generation_config from openai/whisper-small so language/task
        # and suppress_tokens behave identically without conflicting max_length warnings.
        base_gen_cfg = GenerationConfig.from_pretrained("openai/whisper-small", token=token)
        self.model.generation_config = base_gen_cfg
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="gu", task="transcribe"
        )
        self.model.config.forced_decoder_ids = self.forced_decoder_ids
        self.model.generation_config.forced_decoder_ids = self.forced_decoder_ids
        self.model.generation_config.max_length = 448

    @staticmethod
    def _split_audio(audio_array: np.ndarray) -> list[np.ndarray]:
        max_seconds = 29.5
        overlap_seconds = 0.5
        max_samples = int(max_seconds * TARGET_SR)
        step = int((max_seconds - overlap_seconds) * TARGET_SR)
        if len(audio_array) <= max_samples:
            return [audio_array]
        chunks = []
        start = 0
        while start < len(audio_array):
            chunk = audio_array[start:start + max_samples]
            if len(chunk) >= int(0.25 * TARGET_SR):
                chunks.append(chunk)
            if start + max_samples >= len(audio_array):
                break
            start += step
        return chunks or [audio_array[:max_samples]]

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

    def _generate_chunks_batch(self, chunks: Sequence[np.ndarray]) -> list[str]:
        inputs = self.processor.feature_extractor(
            list(chunks),
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            pred_ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                max_new_tokens=self.MAX_NEW_TOKENS,
                num_beams=1,
                do_sample=False,
            )
        decoded = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        return [str(t).strip() for t in decoded]

    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        chunks = self._split_audio(audio_array)
        parts = self._generate_chunks_batch(chunks)
        return self._join_overlapping_transcripts(parts)

    def transcribe_batch(self, audio_arrays, row_indices, batch_size):
        if not audio_arrays:
            return []
        refs: list[tuple[int, int]] = []
        chunks: list[np.ndarray] = []
        per_sample: list[list[str | None]] = []
        errors: list[Exception | None] = [None] * len(audio_arrays)

        for sample_i, audio in enumerate(audio_arrays):
            sample_chunks = self._split_audio(audio)
            per_sample.append([None] * len(sample_chunks))
            for chunk_i, chunk in enumerate(sample_chunks):
                refs.append((sample_i, chunk_i))
                chunks.append(chunk)

        eff_bs = max(1, batch_size)
        for start in range(0, len(chunks), eff_bs):
            chunk_group = chunks[start:start + eff_bs]
            ref_group = refs[start:start + eff_bs]
            try:
                texts = self._generate_chunks_batch(chunk_group)
                for (sample_i, chunk_i), text in zip(ref_group, texts):
                    per_sample[sample_i][chunk_i] = text
            except Exception as batch_exc:
                print(f"\n[WARN] Whisper batch failed; retrying item-by-item: {batch_exc!r}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for (sample_i, chunk_i), chunk in zip(ref_group, chunk_group):
                    try:
                        per_sample[sample_i][chunk_i] = self._generate_chunks_batch([chunk])[0]
                    except Exception as exc:
                        errors[sample_i] = errors[sample_i] or exc

        outputs: list[str | Exception] = []
        for sample_i, parts in enumerate(per_sample):
            if errors[sample_i] is not None:
                outputs.append(errors[sample_i])
            elif any(p is None for p in parts):
                outputs.append(RuntimeError("Whisper failed to produce all chunks"))
            else:
                outputs.append(self._join_overlapping_transcripts([str(p) for p in parts]))
        return outputs


class Gemma4Adapter(BaseASRAdapter):
    batch_mode = "multimodal_batch"
    PROMPT = (
        "Transcribe the following speech segment in Gujarati script. "
        "Output only the verbatim Gujarati transcription, with no explanation."
    )

    def __init__(self):
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        self.name = "gemma-4-E4B-it"
        print(f"[Model] Loading {self.name}...")
        token = hf_auth_token()
        self.processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it", token=token)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            "google/gemma-4-E4B-it", torch_dtype=torch.bfloat16,
            device_map="auto", token=token,
        ).eval()
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    def _messages(self, chunk: np.ndarray):
        return [{"role": "user", "content": [
            {"type": "audio", "audio": chunk, "sampling_rate": TARGET_SR},
            {"type": "text", "text": self.PROMPT},
        ]}]

    def _transcribe_chunk(self, chunk: np.ndarray) -> str:
        inputs = self.processor.apply_chat_template(
            self._messages(chunk), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

    def _transcribe_chunks_batch(self, chunks: Sequence[np.ndarray]) -> list[str]:
        conversations = [self._messages(chunk) for chunk in chunks]
        try:
            inputs = self.processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
                processor_kwargs={"padding": True},
            ).to(self.model.device)
        except TypeError:
            inputs = self.processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
            ).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_len:]
        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        else:
            decoded = self.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(decoded) != len(chunks):
            raise RuntimeError(f"Gemma batch returned {len(decoded)} outputs for {len(chunks)} chunks")
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

    @staticmethod
    def _split_audio(audio_array: np.ndarray) -> list[np.ndarray]:
        max_seconds = 28.0
        overlap_seconds = 0.75
        max_samples = int(max_seconds * TARGET_SR)
        step = int((max_seconds - overlap_seconds) * TARGET_SR)
        if len(audio_array) <= max_samples:
            return [audio_array]
        chunks = []
        start = 0
        while start < len(audio_array):
            chunk = audio_array[start:start + max_samples]
            if len(chunk):
                chunks.append(chunk)
            if start + max_samples >= len(audio_array):
                break
            start += step
        return chunks

    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        return self._join_overlapping_transcripts(
            [self._transcribe_chunk(chunk) for chunk in self._split_audio(audio_array)]
        )

    def transcribe_batch(self, audio_arrays, row_indices, batch_size):
        refs: list[tuple[int, int]] = []
        chunks: list[np.ndarray] = []
        per_sample: list[list[str | None]] = []
        errors: list[Exception | None] = [None] * len(audio_arrays)
        for sample_i, audio in enumerate(audio_arrays):
            sample_chunks = self._split_audio(audio)
            per_sample.append([None] * len(sample_chunks))
            for chunk_i, chunk in enumerate(sample_chunks):
                refs.append((sample_i, chunk_i))
                chunks.append(chunk)

        chunk_batch_size = max(1, min(batch_size, 8))
        for start in range(0, len(chunks), chunk_batch_size):
            chunk_group = chunks[start:start + chunk_batch_size]
            ref_group = refs[start:start + chunk_batch_size]
            try:
                texts = self._transcribe_chunks_batch(chunk_group)
                for (sample_i, chunk_i), text in zip(ref_group, texts):
                    per_sample[sample_i][chunk_i] = text
            except Exception as batch_exc:
                print(f"\n[WARN] Gemma batch failed; retrying chunk group individually: {batch_exc!r}")
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
                outputs.append(RuntimeError("Gemma did not produce every audio chunk"))
            else:
                outputs.append(self._join_overlapping_transcripts([str(p) for p in parts]))
        return outputs


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
    batch_mode = "multi_file_batch_with_fallback"
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
        fd, tmp_path = tempfile.mkstemp(prefix="sravaani_", suffix=".wav", dir=SHM_DIR)
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

    def transcribe_batch(self, audio_arrays, row_indices, batch_size):
        tmp_paths = []
        try:
            for audio in audio_arrays:
                fd, tmp_path = tempfile.mkstemp(prefix="sravaani_batch_", suffix=".wav", dir=SHM_DIR)
                os.close(fd)
                sf.write(tmp_path, audio, TARGET_SR, subtype="PCM_16")
                tmp_paths.append(tmp_path)
            try:
                hypotheses = self.model.transcribe(tmp_paths, return_hypotheses=True)
                if not isinstance(hypotheses, (list, tuple)) or len(hypotheses) != len(tmp_paths):
                    raise RuntimeError(
                        f"SraVaani returned {len(hypotheses) if hasattr(hypotheses, '__len__') else 'unknown'} "
                        f"hypotheses for {len(tmp_paths)} paths"
                    )
                outputs = []
                for hypothesis in hypotheses:
                    first = hypothesis[0] if isinstance(hypothesis, (list, tuple)) and hypothesis else hypothesis
                    outputs.append(str(getattr(first, "text", first)).strip())
                return outputs
            except Exception as batch_exc:
                print(f"\n[WARN] SraVaani multi-file transcription failed; retrying item-by-item: {batch_exc!r}")
                outputs = []
                for path in tmp_paths:
                    try:
                        hypotheses = self.model.transcribe(path, return_hypotheses=True)
                        if not hypotheses:
                            outputs.append("")
                        else:
                            first = hypotheses[0] if isinstance(hypotheses, (list, tuple)) else hypotheses
                            outputs.append(str(getattr(first, "text", first)).strip())
                    except Exception as exc:
                        outputs.append(exc)
                return outputs
        finally:
            for path in tmp_paths:
                if os.path.exists(path):
                    os.remove(path)


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
        fd, tmp_path = tempfile.mkstemp(prefix="bodhan_", suffix=".wav", dir=SHM_DIR)
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


def log_inference_failure(item: dict[str, Any], model_name: str, error: Exception, failure_path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": item["dataset"],
        "split": item["split"],
        "sample_id": item["sample_id"],
        "row_index": item["row_index"],
        "model": model_name,
        "error": repr(error),
    }
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_path.open("a", encoding="utf-8") as handle:
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
        expected = EXPECTED_ROWS.get(dataset)
        coverage = f"{m['n']}/{expected}" if expected else str(m['n'])
        table.append([
            f"{dataset} ({m['split']})", model, coverage,
            f"{m['corpus_wer']:.2f}%", f"{m['corpus_cer']:.2f}%",
            f"{m['mean_capped_cer']:.2f}%", f"{m['drift_pct']:.1f}%",
            f"{m['loop_pct']:.1f}%", f"{m['median_rtf']:.4f}",
        ])
    print(tabulate(table, headers=[
        "Dataset", "Model", "N / Expected", "Corpus WER", "Corpus CER",
        "Mean Capped CER", "Drift %", "Loop %", "Median RTF",
    ], tablefmt="github"))

    models = sorted({model for _, model in metrics})
    print("\n" + "=" * 112)
    print("2. COVERAGE (PARTIAL DATASETS ARE NOT TREATED AS COMPLETE)")
    print("=" * 112)
    coverage_table = []
    for model in models:
        complete, incomplete, missing = [], [], []
        for dataset in expected_datasets:
            entry = metrics.get((dataset, model))
            expected = EXPECTED_ROWS.get(dataset)
            if entry is None:
                missing.append(dataset)
            elif expected is not None and entry["n"] >= expected:
                complete.append(f"{dataset} {entry['n']}/{expected}")
            else:
                incomplete.append(f"{dataset} {entry['n']}/{expected or '?'}")
        coverage_table.append([model, ", ".join(complete) or "—",
                               ", ".join(incomplete) or "—", ", ".join(missing) or "—"])
    print(tabulate(coverage_table, headers=["Model", "Complete", "Partial", "Missing"], tablefmt="github"))

    print("\n" + "=" * 112)
    print("3. THREE-DATASET MACRO / POOLED LEADERBOARD (FULL COVERAGE ONLY)")
    print("=" * 112)
    macro_table = []
    for model in models:
        if not all((d, model) in metrics and metrics[(d, model)]["n"] >= EXPECTED_ROWS.get(d, 10**18)
                   for d in expected_datasets):
            continue
        per_dataset = [metrics[(d, model)] for d in expected_datasets]
        macro_wer = float(np.mean([m["corpus_wer"] for m in per_dataset]))
        macro_cer = float(np.mean([m["corpus_cer"] for m in per_dataset]))
        all_refs = [ref for m in per_dataset for ref in m["refs"]]
        all_hyps = [hyp for m in per_dataset for hyp in m["hyps"]]
        pooled_wer = 100.0 * corpus_wer(all_refs, all_hyps) if all_refs else 0.0
        pooled_cer = 100.0 * corpus_cer(all_refs, all_hyps) if all_refs else 0.0
        macro_table.append([model, sum(m["n"] for m in per_dataset),
                            f"{macro_wer:.2f}%", f"{macro_cer:.2f}%",
                            f"{pooled_wer:.2f}%", f"{pooled_cer:.2f}%"])
    if macro_table:
        macro_table.sort(key=lambda r: float(r[3].rstrip("%")))
        print(tabulate(macro_table, headers=["Model", "Total N", "Macro WER", "Macro CER", "Pooled WER", "Pooled CER"], tablefmt="github"))
    else:
        print("No model has full row coverage across all selected datasets yet.")


def _existing_sample_ids(conn: sqlite3.Connection, dataset: str, split: str, model_name: str) -> set[str]:
    rows = conn.execute(
        "SELECT sample_id FROM predictions WHERE dataset=? AND split=? AND model_name=?",
        (dataset, split, model_name),
    ).fetchall()
    return {row[0] for row in rows}


def _write_performance(record: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with PERFORMANCE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _batch_values(conn, batch_items, outputs, model_name, failure_path, inference_elapsed):
    if len(outputs) != len(batch_items):
        raise RuntimeError(f"Adapter returned {len(outputs)} outputs for {len(batch_items)} samples")
    total_duration = sum(item["duration_s"] for item in batch_items)
    amortized_latency = inference_elapsed / max(len(batch_items), 1)
    batch_rtf = inference_elapsed / max(total_duration, 0.1)
    rows = []
    failures = 0
    for item, output in zip(batch_items, outputs):
        if isinstance(output, Exception):
            failures += 1
            log_inference_failure(item, model_name, output, failure_path)
            continue
        hypothesis = str(output or "").strip()
        metrics = compute_utterance_metrics(item["reference"], hypothesis)
        rows.append((
            item["dataset"], item["split"], item["row_index"], item["sample_id"],
            model_name, item["duration_s"], item["reference"], hypothesis,
            metrics["ref_norm"], metrics["hyp_norm"], metrics["wer"], metrics["cer"],
            metrics["capped_cer"], round(amortized_latency, 4), round(batch_rtf, 4),
            metrics["script_drift"], metrics["drift_ratio"], metrics["loop_flag"], "",
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO predictions
        (dataset, split, row_index, sample_id, model_name, audio_duration_s,
         reference_raw, hypothesis_raw, reference_norm, hypothesis_norm, wer, cer,
         capped_cer, latency_s, rtf, script_drift, drift_ratio, loop_flag, failure_tag)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return len(rows), failures


def _markdown_table(headers, rows):
    def esc(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(esc(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def generate_artifacts(conn: sqlite3.Connection, datasets: list[str], db_path: Path) -> None:
    """Write CSV/JSON/Markdown and CER/WER/performance PNG plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _dataset_metrics(conn)
    model_names = sorted({model for _, model in metrics})
    leaderboard_rows = []
    for (dataset, model), m in sorted(metrics.items()):
        expected = EXPECTED_ROWS.get(dataset)
        leaderboard_rows.append({
            "dataset": dataset, "split": m["split"], "model": model,
            "n": m["n"], "expected_n": expected,
            "coverage_pct": (100.0 * m["n"] / expected) if expected else None,
            "corpus_wer_pct": m["corpus_wer"], "corpus_cer_pct": m["corpus_cer"],
            "mean_capped_cer_pct": m["mean_capped_cer"],
            "median_rtf": m["median_rtf"], "script_drift_pct": m["drift_pct"],
            "loop_pct": m["loop_pct"],
        })
    csv_path = RESULTS_DIR / "leaderboard_optimized.csv"
    fieldnames = list(leaderboard_rows[0].keys()) if leaderboard_rows else [
        "dataset", "split", "model", "n", "expected_n", "coverage_pct",
        "corpus_wer_pct", "corpus_cer_pct", "mean_capped_cer_pct", "median_rtf",
        "script_drift_pct", "loop_pct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(leaderboard_rows)

    perf_records = []
    if PERFORMANCE_PATH.exists():
        for line in PERFORMANCE_PATH.read_text(encoding="utf-8").splitlines():
            try: perf_records.append(json.loads(line))
            except json.JSONDecodeError: pass
    perf_csv = RESULTS_DIR / "performance_optimized.csv"
    perf_fields = sorted({key for row in perf_records for key in row}) or [
        "model", "dataset", "samples", "wall_s", "inference_s", "samples_per_inference_s",
        "audio_seconds_per_inference_second", "batch_size", "cpu_workers", "peak_vram_gb",
    ]
    with perf_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader(); writer.writerows(perf_records)

    summary_json = {
        "database": str(db_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "leaderboard": leaderboard_rows,
        "performance_runs": perf_records,
    }
    (RESULTS_DIR / "benchmark_summary_optimized.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not generate PNG plots (install matplotlib): {exc!r}")
        plt = None

    if plt is not None and leaderboard_rows:
        charts_dir = RESULTS_DIR / "charts_optimized"
        charts_dir.mkdir(parents=True, exist_ok=True)
        for metric_key, metric_title, filename in [
            ("corpus_cer_pct", "Gujarati ASR — Corpus CER by dataset", "cer_by_dataset.png"),
            ("corpus_wer_pct", "Gujarati ASR — Corpus WER by dataset", "wer_by_dataset.png"),
        ]:
            available_datasets = [d for d in datasets if any(r["dataset"] == d for r in leaderboard_rows)]
            fig, axes = plt.subplots(1, max(1, len(available_datasets)),
                                     figsize=(max(7, 5 * len(available_datasets)), 5), squeeze=False)
            for ax, dataset in zip(axes[0], available_datasets):
                rows = [r for r in leaderboard_rows if r["dataset"] == dataset]
                rows.sort(key=lambda r: r[metric_key])
                labels = [r["model"] for r in rows]
                values = [r[metric_key] for r in rows]
                bars = ax.bar(range(len(values)), values)
                ax.set_title(f"{dataset} (n per model shown in CSV)")
                ax.set_ylabel("Error rate (%)")
                ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
                ax.grid(axis="y", alpha=0.25)
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.1f}",
                            ha="center", va="bottom", fontsize=8)
            fig.suptitle(metric_title)
            fig.tight_layout()
            fig.savefig(charts_dir / filename, dpi=180, bbox_inches="tight")
            plt.close(fig)

        grouped = defaultdict(lambda: {"samples": 0, "inference_s": 0.0, "audio_s": 0.0})
        for row in perf_records:
            key = (row.get("model", "unknown"), row.get("dataset", "unknown"))
            grouped[key]["samples"] += int(row.get("samples", 0))
            grouped[key]["inference_s"] += float(row.get("inference_s", 0.0))
            grouped[key]["audio_s"] += float(row.get("audio_seconds", 0.0))
        if grouped:
            labels, rates = [], []
            for (model, dataset), vals in sorted(grouped.items()):
                if vals["inference_s"] > 0:
                    labels.append(f"{model}\n{dataset}")
                    rates.append(vals["samples"] / vals["inference_s"])
            fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.15), 5))
            ax.bar(range(len(labels)), rates)
            ax.set_title("Inference throughput (aggregated across runs)")
            ax.set_ylabel("Samples / inference second")
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(charts_dir / "throughput_samples_per_second.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    report_lines = [
        "# Gujarati ASR benchmark report (optimized runner)", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Database: `{db_path}`", "",
        "## Evaluation protocol", "",
        "- FLEURS Gujarati **test** split; IndicVoices Gujarati **valid** split; Kathbath Gujarati **valid** split.",
        "- Corpus WER/CER are computed on normalized reference/hypothesis text.",
        "- Macro scores should only be compared when a model has complete coverage of all selected datasets.",
        "- Whisper uses direct SDPA batched generation with `max_new_tokens=440` to avoid Gujarati BPE truncation.", "",
        "## Per-dataset leaderboard", "",
    ]
    report_rows = []
    for r in leaderboard_rows:
        cov = "n/a" if r["coverage_pct"] is None else f"{r['coverage_pct']:.1f}%"
        report_rows.append([r["dataset"], r["model"], r["n"], cov,
                            f"{r['corpus_wer_pct']:.2f}%", f"{r['corpus_cer_pct']:.2f}%",
                            f"{r['mean_capped_cer_pct']:.2f}%", f"{r['median_rtf']:.4f}"])
    report_lines.append(_markdown_table(
        ["Dataset", "Model", "N", "Coverage", "Corpus WER", "Corpus CER", "Mean capped CER", "Median RTF"],
        report_rows,
    ))
    report_lines += ["", "## Complete-coverage macro / pooled scores", ""]
    macro_rows = []
    for model in model_names:
        if all((d, model) in metrics and metrics[(d, model)]["n"] >= EXPECTED_ROWS.get(d, 10**18)
               for d in datasets):
            dm = [metrics[(d, model)] for d in datasets]
            refs = [x for m in dm for x in m["refs"]]
            hyps = [x for m in dm for x in m["hyps"]]
            macro_rows.append([model, sum(m["n"] for m in dm),
                               f"{np.mean([m['corpus_wer'] for m in dm]):.2f}%",
                               f"{np.mean([m['corpus_cer'] for m in dm]):.2f}%",
                               f"{100*corpus_wer(refs, hyps):.2f}%",
                               f"{100*corpus_cer(refs, hyps):.2f}%"])
    if macro_rows:
        report_lines.append(_markdown_table(["Model", "Total N", "Macro WER", "Macro CER", "Pooled WER", "Pooled CER"], macro_rows))
    else:
        report_lines.append("No model has full row coverage on every selected dataset; macro scores are intentionally withheld.")
    report_lines += ["", "## Throughput observations", ""]
    if perf_records:
        perf_rows = []
        for row in perf_records:
            perf_rows.append([
                row.get("model", ""), row.get("dataset", ""), row.get("samples", 0),
                f"{float(row.get('samples_per_inference_s', 0)):.3f}",
                f"{float(row.get('audio_seconds_per_inference_second', 0)):.3f}",
                row.get("batch_size", ""), row.get("batch_mode", ""), row.get("cpu_workers", ""),
                f"{float(row.get('peak_vram_gb', 0)):.2f}",
            ])
        report_lines.append(_markdown_table(
            ["Model", "Dataset", "Samples", "Samples/s", "Audio s/s", "Batch", "Batch mode", "CPU workers", "Peak VRAM GB"],
            perf_rows,
        ))
    else:
        report_lines.append("No performance records found yet.")
    (RESULTS_DIR / "benchmark_report_optimized.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[Artifacts] Wrote {csv_path}, {perf_csv}, benchmark_report_optimized.md and charts_optimized/")


def compare_prediction_databases(reference_db: Path, current_db: Path) -> Path | None:
    if not reference_db.exists():
        print(f"[WARN] Comparison DB does not exist: {reference_db}")
        return None
    old = sqlite3.connect(str(reference_db))
    new = sqlite3.connect(str(current_db))
    try:
        old_rows = old.execute(
            "SELECT dataset, split, sample_id, model_name, hypothesis_raw FROM predictions"
        ).fetchall()
        new_rows = new.execute(
            "SELECT dataset, split, sample_id, model_name, hypothesis_raw FROM predictions"
        ).fetchall()
    finally:
        old.close(); new.close()
    old_map = {(r[0], r[1], r[2], r[3]): (r[4] or "") for r in old_rows}
    new_map = {(r[0], r[1], r[2], r[3]): (r[4] or "") for r in new_rows}
    common = sorted(set(old_map) & set(new_map))
    if not common:
        print("[Compare] No overlapping samples between reference and current DB.")
        return None
    by_model = defaultdict(list)
    for key in common:
        by_model[key[3]].append((old_map[key], new_map[key]))
    records = []
    batchable_models = {"sravaani-1.0", "gemma-4-E4B-it", "whisper-gujarati-small", "whisper-small"}
    for model, pairs in sorted(by_model.items()):
        if model not in batchable_models:
            continue
        exact = sum(a == b for a, b in pairs)
        old_hyps = [a for a, _ in pairs]
        new_hyps = [b for _, b in pairs]
        transcript_cer = 100.0 * corpus_cer(old_hyps, new_hyps) if pairs else 0.0
        records.append({
            "model": model,
            "overlap_n": len(pairs),
            "exact_match_n": exact,
            "exact_match_pct": 100.0 * exact / len(pairs),
            "CER_between_old_and_new_hypotheses_pct": transcript_cer,
        })
    if not records:
        return None
    out = RESULTS_DIR / "batch_equivalence.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader(); writer.writerows(records)
    return out


def run_one_dataset(conn, adapter, dataset_key, ds_info, args, failure_path):
    ds, split_name, audio_column, text_column = ds_info
    already_done = _existing_sample_ids(conn, dataset_key, split_name, adapter.name)
    n_limit = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)
    pending_expected = max(0, n_limit - len(already_done.intersection(
        {f"{dataset_key}_{i:06d}" for i in range(n_limit)})))
    print(f"[Run] {adapter.name} / {dataset_key}: rows={n_limit}, already_done={len(already_done)}, pending~={pending_expected}, batch={args.batch_size}, decode_workers={args.workers}")

    if pending_expected == 0:
        print(f"[Done] {adapter.name}/{dataset_key}: nothing pending; preserved predictions reused.")
        return

    iterator = iter_preprocessed_items(
        ds, dataset_key, split_name, audio_column, text_column,
        args.max_samples, already_done, args.workers, args.prefetch,
    )
    pbar = tqdm(desc=f"{adapter.name} on {dataset_key}", total=pending_expected, unit="utt")
    new_count = failed_count = 0
    inference_elapsed = 0.0
    wall_start = time.perf_counter()
    audio_seconds = 0.0
    batch_count = 0

    def process_batch(items):
        nonlocal new_count, failed_count, inference_elapsed, audio_seconds, batch_count
        if not items:
            return
        arrays = [item["audio_array"] for item in items]
        row_indices = [item["row_index"] for item in items]
        _synchronize_cuda()
        t0 = time.perf_counter()
        outputs = adapter.transcribe_batch(arrays, row_indices, args.batch_size)
        _synchronize_cuda()
        elapsed = time.perf_counter() - t0
        inference_elapsed += elapsed
        audio_seconds += sum(item["duration_s"] for item in items)
        batch_count += 1
        successes, failures = _batch_values(conn, items, outputs, adapter.name, failure_path, elapsed)
        new_count += successes
        failed_count += failures
        pbar.update(len(items))

    bucket = []
    bucket_window = args.batch_size * args.bucket_multiplier
    try:
        for item in iterator:
            bucket.append(item)
            if len(bucket) >= bucket_window:
                bucket.sort(key=lambda x: x["duration_s"])
                for offset in range(0, len(bucket), args.batch_size):
                    process_batch(bucket[offset:offset + args.batch_size])
                bucket = []
        if bucket:
            bucket.sort(key=lambda x: x["duration_s"])
            for offset in range(0, len(bucket), args.batch_size):
                process_batch(bucket[offset:offset + args.batch_size])
    finally:
        pbar.close()

    wall_elapsed = time.perf_counter() - wall_start
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    record = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
        "model": adapter.name,
        "dataset": dataset_key,
        "samples": new_count,
        "inference_failures": failed_count,
        "batch_count": batch_count,
        "batch_size": args.batch_size if adapter.batch_mode != "serial_fallback" else 1,
        "batch_mode": adapter.batch_mode,
        "cpu_workers": args.workers,
        "prefetch": args.prefetch,
        "bucket_multiplier": args.bucket_multiplier,
        "wall_s": round(wall_elapsed, 3),
        "inference_s": round(inference_elapsed, 3),
        "audio_seconds": round(audio_seconds, 3),
        "samples_per_inference_s": round(new_count / max(inference_elapsed, 1e-9), 4),
        "audio_seconds_per_inference_second": round(audio_seconds / max(inference_elapsed, 1e-9), 4),
        "end_to_end_samples_per_s": round(new_count / max(wall_elapsed, 1e-9), 4),
        "peak_vram_gb": round(peak_vram_gb, 3),
    }
    if batch_count > 0:
        _write_performance(record)
        print(f"[Done] {adapter.name}/{dataset_key}: new={new_count}, failures={failed_count}, batches={batch_count}, inference={inference_elapsed/60:.1f} min, throughput={record['samples_per_inference_s']:.3f} samples/s, audio-throughput={record['audio_seconds_per_inference_second']:.3f} audio-s/s")
    else:
        print(f"[Done] {adapter.name}/{dataset_key}: nothing pending; preserved predictions reused.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel/batched Gujarati ASR benchmark evaluator")
    parser.add_argument("--datasets", nargs="+", default=["fleurs", "indicvoices", "kathbath"], choices=list(DATASET_SPLITS))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_KEYS, choices=MODEL_KEYS)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 evaluates the full split; positive values limit rows for calibration/smoke tests")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="microbatch size for Whisper (Gemma auto-caps at 8)")
    parser.add_argument("--workers", type=int, default=8, help="parallel CPU audio decode/resample workers")
    parser.add_argument("--prefetch", type=int, default=32, help="maximum queued audio decode tasks")
    parser.add_argument("--bucket-multiplier", type=int, default=4, help="sort this many batches by duration to reduce padding")
    parser.add_argument("--torch-threads", type=int, default=4, help="PyTorch CPU intra-op threads")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--failure-path", type=Path, default=DEFAULT_FAILURE_PATH)
    parser.add_argument("--compare-db", type=Path, default=None, help="optional preserved serial DB for output-equivalence comparison")
    parser.add_argument("--recompute-models", nargs="*", default=[], choices=MODEL_KEYS,
                        help="delete these models' existing predictions in this DB before running them")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.max_samples < 0 or args.batch_size < 1 or args.workers < 1 or args.prefetch < 1 or args.bucket_multiplier < 1 or args.torch_threads < 1:
        parser.error("max-samples must be >=0; batch-size, workers, prefetch and torch-threads must be >=1")

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[Hardware] {torch.cuda.get_device_name(0)} | VRAM={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB | BF16={torch.cuda.is_bf16_supported()}")

    conn = init_db(args.db_path)
    try:
        if args.summary_only:
            print_leaderboard(conn, args.datasets)
            generate_artifacts(conn, args.datasets, args.db_path)
            if args.compare_db is not None:
                compare_prediction_databases(args.compare_db, args.db_path)
            return 0

        if args.recompute_models:
            for model_key in args.recompute_models:
                model_name = MODEL_NAME_BY_KEY[model_key]
                conn.execute("DELETE FROM predictions WHERE model_name=? AND dataset IN (%s)" % ",".join("?" for _ in args.datasets),
                             [model_name, *args.datasets])
                conn.commit()
                print(f"[DB] Recomputed model={model_name}: deleted old rows for datasets={args.datasets}")

        dataset_cache = {}
        for dataset_key in args.datasets:
            dataset_cache[dataset_key] = load_gujarati_dataset(dataset_key)

        init_failures = []
        for model_key in args.models:
            # Skip initializing heavy models if all requested datasets are already complete in SQLite
            model_name = MODEL_NAME_BY_KEY[model_key]
            if args.max_samples == 0 and all(
                len(_existing_sample_ids(conn, d, DATASET_SPLITS[d], model_name)) >= EXPECTED_ROWS[d]
                for d in args.datasets
            ):
                print(f"[Skip] {model_name} already has 100% full coverage across {args.datasets}; skipping model load.")
                continue

            try:
                adapter = build_adapter(model_key)
            except Exception as exc:
                print(f"[ERROR] Could not initialize model '{model_key}': {exc!r}")
                init_failures.append((model_key, repr(exc)))
                continue

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for dataset_key in args.datasets:
                run_one_dataset(conn, adapter, dataset_key, dataset_cache[dataset_key], args, args.failure_path)

            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print_leaderboard(conn, args.datasets)
        if init_failures:
            print("\n[Model initialization failures]")
            for model, error in init_failures:
                print(f"- {model}: {error}")
        generate_artifacts(conn, args.datasets, args.db_path)
        if args.compare_db is not None:
            compare_prediction_databases(args.compare_db, args.db_path)
        return 0 if not init_failures else 2
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
