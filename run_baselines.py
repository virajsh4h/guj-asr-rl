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

class BaseASRAdapter:
    name: str = "base"
    def transcribe(self, audio_array: np.ndarray, row_index: int) -> str:
        raise NotImplementedError

class WhisperSmallAdapter(BaseASRAdapter):
    def __init__(self):
        from transformers import pipeline
        self.name = "whisper-small"
        print(f"[Model] Loading {self.name}...")
        self.pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.bfloat16)

    def transcribe(self, audio_array, row_index):
        out = self.pipe({"array": audio_array, "sampling_rate": 16000}, generate_kwargs={"language": "gujarati", "task": "transcribe"})
        return out["text"].strip()

class Gemma4Adapter(BaseASRAdapter):
    def __init__(self):
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        self.name = "gemma-4-E4B-it"
        print(f"[Model] Loading {self.name}...")
        self.processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
        self.model = AutoModelForMultimodalLM.from_pretrained("google/gemma-4-E4B-it", torch_dtype=torch.bfloat16, device_map="auto").eval()

    def transcribe(self, audio_array, row_index):
        messages = [{"role": "user", "content": [{"type": "audio", "audio": audio_array, "sampling_rate": 16000}, {"type": "text", "text": "Transcribe the following speech segment in Gujarati script. Output only the raw Gujarati transcription with no extra commentary."}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=224, do_sample=False)
        return self.processor.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

class IndicConformerAdapter(BaseASRAdapter):
    def __init__(self):
        self.name = "indicconformer-600m"
        self.use_nemo = False
        try:
            from transformers import AutoModel
            print("[Model] Loading IndicConformer HF wrapper...")
            self.model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
            if torch.cuda.is_available(): self.model = self.model.cuda()
        except Exception as e:
            print(f"[Model] HF failed ({e}); falling back to NeMo...")
            import nemo.collections.asr as nemo_asr
            self.use_nemo = True
            self.model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large")
            if torch.cuda.is_available(): self.model = self.model.cuda()
            self.model.eval()

    def transcribe(self, audio_array, row_index):
        if not self.use_nemo:
            wav_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            if torch.cuda.is_available(): wav_tensor = wav_tensor.cuda()
            with torch.inference_mode(): hyp = self.model(wav_tensor, "gu", "rnnt")
            return hyp if isinstance(hyp, str) else str(hyp[0])
        else:
            import soundfile as sf
            tmp = f"/tmp/{row_index}.wav"
            sf.write(tmp, audio_array, 16000)
            with torch.inference_mode(): hyps = self.model.transcribe([tmp], batch_size=1, language_id="gu")
            os.remove(tmp)
            res = hyps[0][0] if isinstance(hyps[0], list) else hyps[0]
            return res.text if hasattr(res, "text") else str(res)

class SraVaaniAdapter(BaseASRAdapter):
    def __init__(self):
        import nemo.collections.asr as nemo_asr
        self.name = "sravaani-1.0"
        print(f"[Model] Loading {self.name}...")
        self.model = nemo_asr.models.ASRModel.from_pretrained("ARTPARK-IISc/SraVaani-1.0")
        if torch.cuda.is_available(): self.model = self.model.cuda().to(torch.bfloat16)
        self.model.eval()

    def transcribe(self, audio_array, row_index):
        import soundfile as sf
        tmp = f"/tmp/{row_index}.wav"
        sf.write(tmp, audio_array, 16000)
        with torch.inference_mode():
            try: hyps = self.model.transcribe([tmp], batch_size=1, source_lang="gu", target_lang="gu")
            except TypeError: hyps = self.model.transcribe([tmp], batch_size=1)
        os.remove(tmp)
        res = hyps[0][0] if isinstance(hyps[0], list) else hyps[0]
        return res.text.strip() if hasattr(res, "text") else str(res).strip()

class BodhanAdapter(BaseASRAdapter):
    def __init__(self):
        from huggingface_hub import snapshot_download
        print("[Model] Downloading BodhanAI custom package...")
        model_dir = snapshot_download("bodhan-ai/indic-transcribe-core")
        sys.path.insert(0, model_dir)
        from indic_transcribe import IndicTranscribe
        self.name = "indic-transcribe-core"
        self.asr = IndicTranscribe.from_pretrained(model_dir)

    def transcribe(self, audio_array, row_index):
        import soundfile as sf
        tmp = f"/tmp/{row_index}.wav"
        sf.write(tmp, audio_array, 16000)
        try: return self.asr(tmp, lang="gu").strip()
        finally: os.remove(tmp)

def build_adapter(key):
    if key == "whisper": return WhisperSmallAdapter()
    if key == "gemma4": return Gemma4Adapter()
    if key == "indicconformer": return IndicConformerAdapter()
    if key == "sravaani": return SraVaaniAdapter()
    if key == "bodhan": return BodhanAdapter()
    raise ValueError(key)
