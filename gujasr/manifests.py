from __future__ import annotations

DATASETS = {
    "fleurs": {
        "repo": "google/fleurs",
        "config": "gu_in",
        "split": "test",
        "revision": "fdd74f7cdf01fc7641ab3072b6a38d364b3667f6",
        "expected_rows": 1000,
        "domain": "read speech, FLoRes domain",
    },
    "indicvoices": {
        "repo": "ai4bharat/IndicVoices",
        "config": "gujarati",
        "split": "valid",
        "revision": "c96f9088f138cf89d419da7e8e643e1f05c00a87",
        "expected_rows": 2968,
        "domain": "spontaneous, crowd-sourced, varied recording conditions",
    },
    "kathbath": {
        "repo": "ai4bharat/Kathbath",
        "config": "gujarati",
        "split": "valid",
        "revision": "5b9e92849222026d9141acba4e8434fe816396bf",
        "expected_rows": 2905,
        "domain": "narrative and conversational prompts",
    },
}

MODELS = {
    "bodhan": {
        "name": "indic-transcribe-core",
        "family": "Indic ASR, CTC",
        "note": "AI4Bharat Indic Transcribe Core checkpoint",
    },
    "sravaani": {
        "name": "sravaani-1.0",
        "family": "Indic ASR",
        "note": "SraVaani release, Gujarati supported",
    },
    "whisper-gu": {
        "name": "whisper-gujarati-small",
        "family": "Whisper small, Gujarati fine-tune",
        "note": "vasista22/whisper-gujarati-small",
    },
    "whisper": {
        "name": "whisper-small",
        "family": "Whisper small, multilingual",
        "note": "openai/whisper-small, no Gujarati fine-tune",
    },
    "gemma4": {
        "name": "gemma-4-E4B-it",
        "family": "Speech-capable multimodal LLM, base",
        "note": "instruction-tuned baseline, no ASR adaptation",
    },
    "gemma4-step5": {
        "name": "gemma-4-E4B-step5",
        "family": "Speech-capable multimodal LLM, constrained decode",
        "note": "logit-mask stage",
    },
    "gemma4-sft": {
        "name": "gemma-4-E4B-sft",
        "family": "Speech-capable multimodal LLM, SFT",
        "note": "308-layer LoRA-style SFT",
    },
    "gemma4-grpo": {
        "name": "gemma-4-E4B-grpo",
        "family": "Speech-capable multimodal LLM, SFT + GRPO",
        "note": "RL stage on top of SFT",
    },
    "gemma4-ctx": {
        "name": "gemma-4-E4B-ctx",
        "family": "Speech-capable multimodal LLM, SFT + GRPO + context",
        "note": "context-injection adapter",
    },
}

PUBLIC_MODELS = ("bodhan", "sravaani", "whisper-gu", "whisper")


def model_name(model_key: str) -> str:
    return MODELS[model_key]["name"]


def dataset_manifest(dataset_key: str) -> dict:
    spec = dict(DATASETS[dataset_key])
    spec["key"] = dataset_key
    return spec
