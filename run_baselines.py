#!/usr/bin/env python3
import os, sys, time, sqlite3, argparse, tempfile
import numpy as np
import torch
from tqdm import tqdm
from tabulate import tabulate
from datasets import load_dataset, Audio
import pandas as pd
from jiwer import wer as corpus_wer, cer as corpus_cer

from gu_normalizer import compute_utterance_metrics

WORKSPACE = os.path.expanduser("~/gujarati-asr-sota")
DATA_CACHE = os.path.join(WORKSPACE, "data_cache")
DB_PATH = os.path.join(WORKSPACE, "results", "predictions.sqlite")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            dataset TEXT, split TEXT, row_index INTEGER, sample_id TEXT, model_name TEXT,
            audio_duration_s REAL, reference_raw TEXT, hypothesis_raw TEXT,
            reference_norm TEXT, hypothesis_norm TEXT, wer REAL, cer REAL,
            capped_cer REAL, latency_s REAL, rtf REAL, script_drift INTEGER,
            drift_ratio REAL, loop_flag INTEGER, failure_tag TEXT DEFAULT '',
            PRIMARY KEY (dataset, split, sample_id, model_name)
        )
    """)
    conn.commit()
    return conn

def is_evaluated(conn, dataset, split, sample_id, model_name):
    cur = conn.execute("SELECT 1 FROM predictions WHERE dataset=? AND split=? AND sample_id=? AND model_name=?", (dataset, split, sample_id, model_name))
    return cur.fetchone() is not None

def load_gujarati_benchmark(dataset_key, max_samples=0):
    hf_token = os.environ.get("HF_TOKEN", None)
    if dataset_key == "fleurs":
        print("[Dataset] Loading google/fleurs (gu_in, test) via Parquet...")
        ds = load_dataset("google/fleurs", "gu_in", split="test", cache_dir=DATA_CACHE, revision="refs/convert/parquet")
        split_name, text_col, id_col = "test", "transcription", "id"
    elif dataset_key == "indicvoices":
        print("[Dataset] Loading ai4bharat/IndicVoices (gujarati, valid)...")
        ds = load_dataset("ai4bharat/IndicVoices", "gujarati", split="valid", cache_dir=DATA_CACHE, token=hf_token, trust_remote_code=True)
        split_name, text_col = "valid", "text" if "text" in ds.column_names else "normalized"
        id_col = "utt_id" if "utt_id" in ds.column_names else None
    elif dataset_key == "kathbath":
        print("[Dataset] Loading ai4bharat/Kathbath (gujarati, valid)...")
        ds = load_dataset("ai4bharat/Kathbath", "gujarati", split="valid", cache_dir=DATA_CACHE, token=hf_token, trust_remote_code=True)
        split_name, text_col = "valid", "text" if "text" in ds.column_names else "transcription"
        id_col = "utt_id" if "utt_id" in ds.column_names else None
    
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    total = len(ds) if max_samples <= 0 else min(len(ds), max_samples)

    for idx in range(total):
        row = ds[idx]
        sample_id = str(row[id_col]) if (id_col and id_col in row) else f"{dataset_key}_{idx:05d}"
        arr = np.asarray(row["audio"]["array"], dtype=np.float32)
        if arr.ndim > 1: arr = arr.mean(axis=1)
        duration_s = round(len(arr) / 16000.0, 3)
        ref_text = str(row.get(text_col, "")).strip()
        if not ref_text or duration_s < 0.3: continue
        yield {"dataset": dataset_key, "split": split_name, "row_index": idx, "sample_id": sample_id, "audio_array": arr, "duration_s": duration_s, "reference": ref_text}
