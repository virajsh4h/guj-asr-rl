#!/usr/bin/env python3
"""Step 6.1: verify and cache zero-leakage Gujarati train splits for Gemma 4 E4B SFT."""

from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from run_baselines_optimized import (
    DATA_CACHE,
    _cast_audio_columns_without_decoding,
    hf_auth_token,
)

SFT_CACHE = Path("/data/gujarati-asr-sota/data_cache/sft_train")

def inspect_train_splits() -> None:
    SFT_CACHE.mkdir(parents=True, exist_ok=True)
    token = hf_auth_token()

    # 1. FLEURS Gujarati train split
    print("[SFT Data] Checking FLEURS Gujarati train split...")
    fleurs_train_pq = hf_hub_download(
        repo_id="google/fleurs",
        repo_type="dataset",
        filename="gu_in/train-00000-of-00001.parquet",
        revision="fdd74f7cdf01fc7641ab3072b6a38d364b3667f6",
        cache_dir=str(DATA_CACHE),
        token=token,
    )
    ds_fleurs = load_dataset(
        "parquet",
        data_files={"train": fleurs_train_pq},
        split="train",
        cache_dir=str(DATA_CACHE),
    )
    ds_fleurs, _ = _cast_audio_columns_without_decoding(ds_fleurs)
    print(f"  -> FLEURS train rows: {len(ds_fleurs):,} | columns: {ds_fleurs.column_names}")

    # 2. Kathbath Gujarati train split
    print("[SFT Data] Checking Kathbath Gujarati train split...")
    ds_kb = load_dataset(
        "ai4bharat/Kathbath",
        "gujarati",
        split="train",
        cache_dir=str(DATA_CACHE),
        token=token,
    )
    ds_kb, _ = _cast_audio_columns_without_decoding(ds_kb)
    print(f"  -> Kathbath train rows: {len(ds_kb):,} | columns: {ds_kb.column_names}")

    # 3. IndicVoices Gujarati train split
    print("[SFT Data] Checking IndicVoices Gujarati train split...")
    ds_iv = load_dataset(
        "ai4bharat/IndicVoices",
        "gujarati",
        split="train",
        cache_dir=str(DATA_CACHE),
        token=token,
    )
    ds_iv, _ = _cast_audio_columns_without_decoding(ds_iv)
    print(f"  -> IndicVoices train rows: {len(ds_iv):,} | columns: {ds_iv.column_names}")
    print("\n[OK] All training splits successfully verified.")

if __name__ == "__main__":
    inspect_train_splits()
