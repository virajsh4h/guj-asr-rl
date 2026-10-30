#!/usr/bin/env python3
"""
run_baselines.py
Resumable Gujarati ASR evaluation harness across public benchmarks and OSS models.
"""
import os
import io
import sys
import json
import time
import shutil
import sqlite3
import argparse
import importlib.util
import numpy as np
import soundfile as sf
import librosa
import torch
import huggingface_hub

# Shim deprecated symbols removed in huggingface_hub >= 1.0 so NeMo 1.23 imports cleanly
if not hasattr(huggingface_hub, "ModelFilter"):
    huggingface_hub.ModelFilter = object
if not hasattr(huggingface_hub, "DatasetFilter"):
    huggingface_hub.DatasetFilter = object

from tqdm import tqdm
from tabulate import tabulate
from datasets import load_dataset, Audio
import pandas as pd
from huggingface_hub import snapshot_download
from jiwer import wer as corpus_wer, cer as corpus_cer

from gu_normalizer import compute_utterance_metrics

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE = os.path.join(WORKSPACE, "data_cache")
MODEL_CACHE = os.path.join(WORKSPACE, "model_cache")
DB_PATH = os.path.join(WORKSPACE, "results", "predictions.sqlite")


def load_dotenv_if_present():
    env_path = os.path.join(WORKSPACE, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ── 1. SQLite Schema & Resumability ────────────────────────────────────────
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
    cur = conn.execute(
        "SELECT 1 FROM predictions WHERE dataset=? AND split=? AND sample_id=? AND model_name=?",
        (dataset, split, sample_id, model_name),
    )
    return cur.fetchone() is not None


# ── 2. Dataset Loaders (Standardized to 16kHz Mono via soundfile) ──────────
def load_gujarati_benchmark(dataset_key, max_samples=0):
    hf_token = os.environ.get("HF_TOKEN", None)
    os.makedirs(DATA_CACHE, exist_ok=True)

    if dataset_key == "fleurs":
        print("[Dataset] Loading google/fleurs (gu_in, test) via Parquet...")
        try:
            ds = load_dataset(
                "parquet",
                data_files={"test": "hf://datasets/google/fleurs@refs/convert/parquet/gu_in/test/*.parquet"},
                split="test",
                cache_dir=DATA_CACHE,
                token=hf_token,
            )
        except Exception:
            ds = load_dataset(
                "google/fleurs",
                data_files={"test": "gu_in/test/*.parquet"},
                split="test",
                cache_dir=DATA_CACHE,
                revision="refs/convert/parquet",
                token=hf_token,
            )
        split_name, text_col, id_col = "test", "transcription", "id"

    elif dataset_key == "indicvoices":
        print("[Dataset] Loading ai4bharat/IndicVoices (gujarati, valid)...")
        ds = load_dataset(
            "ai4bharat/IndicVoices",
            "gujarati",
            split="valid",
            cache_dir=DATA_CACHE,
            token=hf_token,
        )
        split_name = "valid"
        text_col = "text" if "text" in ds.column_names else "normalized"
        id_col = "utt_id" if "utt_id" in ds.column_names else None

    elif dataset_key == "kathbath":
        print("[Dataset] Loading ai4bharat/Kathbath (gujarati, valid)...")
        ds = load_dataset(
            "ai4bharat/Kathbath",
            "gujarati",
            split="valid",
            cache_dir=DATA_CACHE,
            token=hf_token,
        )
        split_name = "valid"
        text_col = "text" if "text" in ds.column_names else "transcription"
        id_col = "utt_id" if "utt_id" in ds.column_names else None
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    ds = ds.cast_column("audio", Audio(decode=False))
    total = len(ds) if max_samples <= 0 else min(len(ds), max_samples)

    for idx in range(total):
        row = ds[idx]
        sample_id = str(row[id_col]) if (id_col and id_col in row) else f"{dataset_key}_{idx:05d}"

        audio_info = row["audio"]
        if isinstance(audio_info, dict) and audio_info.get("bytes"):
            arr, sr = sf.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
        elif isinstance(audio_info, dict) and audio_info.get("path"):
            arr, sr = sf.read(audio_info["path"], dtype="float32")
        else:
            arr, sr = np.asarray(audio_info["array"], dtype=np.float32), audio_info["sampling_rate"]

        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

        duration_s = round(len(arr) / 16000.0, 3)
        ref_text = str(row.get(text_col, "")).strip()

        if not ref_text or duration_s < 0.3:
            continue

        yield {
            "dataset": dataset_key,
            "split": split_name,
            "row_index": idx,
            "sample_id": sample_id,
            "audio_array": arr,
            "duration_s": duration_s,
            "reference": ref_text,
        }


# ── 3. Modular Model Adapters ──────────────────────────────────────────────
class BaseASRAdapter:
    name: str = "base"

    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        raise NotImplementedError


class WhisperAdapter(BaseASRAdapter):
    def __init__(self, model_id="openai/whisper-small", name="whisper-small"):
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        self.name = name
        print(f"[Model] Loading {self.name} ({model_id})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=self.dtype,
        ).to(self.device).eval()

    def transcribe(self, audio_array, row_index):
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            pred_ids = self.model.generate(
                input_features,
                language="gujarati",
                task="transcribe",
                max_new_tokens=224,
            )
        return self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


class Gemma4Adapter(BaseASRAdapter):
    def __init__(self):
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        self.name = "gemma-4-E4B-it"
        print(f"[Model] Loading {self.name}...")
        self.processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
        self.model = AutoModelForMultimodalLM.from_pretrained(
            "google/gemma-4-E4B-it",
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    def transcribe(self, audio_array, row_index):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array, "sampling_rate": 16000},
                    {
                        "type": "text",
                        "text": "Transcribe the following speech segment in Gujarati script. Output only the raw Gujarati transcription with no extra commentary.",
                    },
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=224, do_sample=False)

        return self.processor.decode(
            out_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()


class IndicConformerAdapter(BaseASRAdapter):
    """Loads IndicConformer-600M directly from materialized local_dir to bypass HF symlink & repo_id bugs."""
    def __init__(self):
        self.name = "indicconformer-600m"
        self.use_nemo = False
        hf_token = os.environ.get("HF_TOKEN", None)
        try:
            local_model_dir = os.path.join(MODEL_CACHE, "indic-conformer-600m-multilingual")
            print(f"[Model] Loading IndicConformer ONNX from {local_model_dir}...")
            snapshot_download(
                repo_id="ai4bharat/indic-conformer-600m-multilingual",
                local_dir=local_model_dir,
                token=hf_token,
            )
            # Ensure no symlinks escape assets/ directory for ONNXRuntime 1.20+ security check
            assets_dir = os.path.join(local_model_dir, "assets")
            if os.path.isdir(assets_dir):
                for fname in os.listdir(assets_dir):
                    fpath = os.path.join(assets_dir, fname)
                    if os.path.islink(fpath):
                        real_target = os.path.realpath(fpath)
                        os.unlink(fpath)
                        shutil.copy2(real_target, fpath)

            with open(os.path.join(local_model_dir, "config.json"), "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)

            spec = importlib.util.spec_from_file_location(
                "indic_model_onnx", os.path.join(local_model_dir, "model_onnx.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            cfg_dict.pop("ts_folder", None)
            config = mod.IndicASRConfig(ts_folder=local_model_dir, **cfg_dict)
            self.model = mod.IndicASRModel(config)
        except Exception as e:
            print(f"[Model] HF ONNX direct load failed ({e}); falling back to NeMo...")
            import nemo.collections.asr as nemo_asr
            self.use_nemo = True
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large"
            )
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()

    def transcribe(self, audio_array, row_index):
        if not self.use_nemo:
            wav_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            with torch.inference_mode():
                hyp = self.model(wav_tensor, "gu", "rnnt")
            return hyp if isinstance(hyp, str) else str(hyp[0])
        else:
            tmp = f"/tmp/ic_{row_index}.wav"
            sf.write(tmp, audio_array, 16000)
            try:
                with torch.inference_mode():
                    hyps = self.model.transcribe([tmp], batch_size=1, language_id="gu")
                res = hyps[0][0] if isinstance(hyps[0], list) else hyps[0]
                return res.text if hasattr(res, "text") else str(res)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)


class SraVaaniAdapter(BaseASRAdapter):
    def __init__(self):
        self.name = "sravaani-1.0"
        hf_token = os.environ.get("HF_TOKEN", None)
        print(f"[Model] Loading {self.name}...")
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                "ARTPARK-IISc/SraVaani-1.0",
                trust_remote_code=True,
                token=hf_token,
            )
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
        except Exception:
            import nemo.collections.asr as nemo_asr
            self.model = nemo_asr.models.ASRModel.from_pretrained("ARTPARK-IISc/SraVaani-1.0")
            if torch.cuda.is_available():
                self.model = self.model.cuda().to(torch.bfloat16)
            self.model.eval()

    def transcribe(self, audio_array, row_index):
        tmp = f"/tmp/sv_{row_index}.wav"
        sf.write(tmp, audio_array, 16000)
        try:
            with torch.inference_mode():
                hyps = self.model.transcribe([tmp], return_hypotheses=True)
            res = hyps[0][0] if isinstance(hyps[0], list) else hyps[0]
            return res.text.strip() if hasattr(res, "text") else str(res).strip()
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


class BodhanAdapter(BaseASRAdapter):
    def __init__(self):
        hf_token = os.environ.get("HF_TOKEN", None)
        print("[Model] Downloading BodhanAI custom package...")
        local_model_dir = os.path.join(MODEL_CACHE, "indic-transcribe-core")
        model_dir = snapshot_download(
            repo_id="bodhan-ai/indic-transcribe-core",
            local_dir=local_model_dir,
            token=hf_token,
        )
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from indic_transcribe import IndicTranscribe
        self.name = "indic-transcribe-core"
        self.asr = IndicTranscribe.from_pretrained(model_dir)

    def transcribe(self, audio_array, row_index):
        tmp = f"/tmp/bodhan_{row_index}.wav"
        sf.write(tmp, audio_array, 16000)
        try:
            return self.asr(tmp, lang="gu").strip()
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def build_adapter(key):
    if key == "whisper":
        return WhisperAdapter("openai/whisper-small", "whisper-small")
    if key == "whisper-gu":
        return WhisperAdapter("vasista22/whisper-gujarati-small", "whisper-gujarati-small")
    if key == "gemma4":
        return Gemma4Adapter()
    if key == "indicconformer":
        return IndicConformerAdapter()
    if key == "sravaani":
        return SraVaaniAdapter()
    if key == "bodhan":
        return BodhanAdapter()
    raise ValueError(f"Unknown model key: {key}")


# ── 4. Leaderboard Summary Printer (Corpus WER/CER + Capped CER) ───────────
def print_leaderboard(conn):
    cur = conn.execute("""
        SELECT dataset, split, model_name,
               reference_norm, hypothesis_norm,
               capped_cer, rtf, script_drift, loop_flag
        FROM predictions
        ORDER BY dataset, model_name
    """)
    rows = cur.fetchall()
    if not rows:
        print("No predictions logged yet.")
        return

    grouped = {}
    for r in rows:
        grouped.setdefault((r[0], r[1], r[2]), []).append(r)

    table = []
    for (dataset, split, model_name), items in grouped.items():
        refs = [x[3] for x in items if x[3]]
        hyps = [x[4] if x[4] else "" for x in items if x[3]]

        c_wer = round(corpus_wer(refs, hyps) * 100.0, 2) if refs else 0.0
        c_cer = round(corpus_cer(refs, hyps) * 100.0, 2) if refs else 0.0
        mean_cap_cer = round(float(np.mean([x[5] for x in items])), 2)
        median_rtf = round(float(np.median([x[6] for x in items])), 4)
        drift_pct = round(100.0 * sum(x[7] for x in items) / len(items), 2)
        loop_pct = round(100.0 * sum(x[8] for x in items) / len(items), 2)

        table.append([
            f"{dataset} ({split})",
            model_name,
            len(items),
            f"{c_wer:.2f}%",
            f"{c_cer:.2f}%",
            f"{mean_cap_cer:.2f}%",
            f"{drift_pct:.1f}%",
            f"{loop_pct:.1f}%",
            median_rtf,
        ])

    headers = [
        "Dataset", "Model", "N", "Corpus WER", "Corpus CER",
        "Mean Capped CER", "Drift %", "Loop %", "Median RTF"
    ]
    print("\n" + "=" * 110)
    print("GUJARATI ASR REPRODUCIBLE BASELINE LEADERBOARD")
    print("=" * 110)
    print(tabulate(table, headers=headers, tablefmt="github"))
    print("=" * 110 + "\n")


# ── 5. Main Execution Loop ─────────────────────────────────────────────────
def main():
    load_dotenv_if_present()
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["fleurs"], choices=["fleurs", "indicvoices", "kathbath"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["whisper", "whisper-gu", "indicconformer", "gemma4"],
        choices=["whisper", "whisper-gu", "indicconformer", "sravaani", "bodhan", "gemma4"],
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    conn = init_db()
    if args.summary_only:
        print_leaderboard(conn)
        return

    for model_key in args.models:
        try:
            adapter = build_adapter(model_key)
        except Exception as e:
            print(f"[ERROR] Could not init {model_key}: {e}")
            continue

        for ds_key in args.datasets:
            try:
                samples = list(load_gujarati_benchmark(ds_key, args.max_samples))
            except Exception as e:
                print(f"[ERROR] Could not load {ds_key}: {e}")
                continue

            pbar = tqdm(samples, desc=f"{adapter.name} on {ds_key}")
            for item in pbar:
                if is_evaluated(conn, item["dataset"], item["split"], item["sample_id"], adapter.name):
                    continue

                t0 = time.perf_counter()
                try:
                    hyp_raw = adapter.transcribe(item["audio_array"], item["row_index"])
                except Exception as inf_err:
                    print(f"\n[WARN] Inference failed on {item['sample_id']}: {inf_err}")
                    hyp_raw = ""

                latency_s = round(time.perf_counter() - t0, 4)
                rtf = round(latency_s / max(item["duration_s"], 0.1), 4)
                m = compute_utterance_metrics(item["reference"], hyp_raw)

                conn.execute("""
                    INSERT OR REPLACE INTO predictions
                    (dataset, split, row_index, sample_id, model_name, audio_duration_s, reference_raw, hypothesis_raw,
                     reference_norm, hypothesis_norm, wer, cer, capped_cer, latency_s, rtf, script_drift, drift_ratio, loop_flag, failure_tag)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    item["dataset"], item["split"], item["row_index"], item["sample_id"],
                    adapter.name, item["duration_s"], item["reference"], hyp_raw,
                    m["ref_norm"], m["hyp_norm"], m["wer"], m["cer"], m["capped_cer"],
                    latency_s, rtf, m["script_drift"], m["drift_ratio"], m["loop_flag"], ""
                ))
                conn.commit()
                pbar.set_postfix(cer=f"{m['cer']:.1f}%", drift=m["script_drift"], loop=m["loop_flag"])

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_leaderboard(conn)


if __name__ == "__main__":
    main()
