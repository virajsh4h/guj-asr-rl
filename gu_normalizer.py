#!/usr/bin/env python3
"""
gu_normalizer.py
Strict Gujarati normalization, script-drift detection, and repetition-loop detection.
"""
import re
import unicodedata
from collections import Counter
from jiwer import wer, cer

try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    _factory = IndicNormalizerFactory()
    _gu_indic_norm = _factory.get_normalizer("gu", remove_nuktas=True)
except Exception:
    _gu_indic_norm = None

GUJARATI_CHAR_RE = re.compile(r"[\u0A80-\u0AFF]")
NON_GUJARATI_ALPHA_RE = re.compile(r"[^\W\d_\u0A80-\u0AFF]", re.UNICODE)
ZW_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
PUNCT_RE = re.compile(r"[^\w\s\u0A80-\u0AFF]", re.UNICODE)
DIGITS_MAP = str.maketrans("૦૧૨૩૪૫૬૭૮૯", "0123456789")

def detect_script_drift(raw_text: str) -> tuple[int, float]:
    if not raw_text or not raw_text.strip():
        return 0, 0.0
    gu_chars = len(GUJARATI_CHAR_RE.findall(raw_text))
    non_gu_chars = len(NON_GUJARATI_ALPHA_RE.findall(raw_text))
    total_alpha = gu_chars + non_gu_chars
    if total_alpha == 0:
        return 0, 0.0
    drift_ratio = non_gu_chars / total_alpha
    return (1 if drift_ratio > 0.05 else 0), round(drift_ratio, 4)

def detect_loop_or_empty(ref_norm: str, hyp_norm: str) -> int:
    if not ref_norm: return 0
    if not hyp_norm: return 1

    ref_len, hyp_len = len(ref_norm), len(hyp_norm)
    if ref_len > 10 and hyp_len > 2.0 * ref_len: return 1
    if ref_len > 15 and hyp_len < 0.2 * ref_len: return 1

    words = hyp_norm.split()
    if len(words) >= 8:
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
        if Counter(bigrams).most_common(1)[0][1] >= 4: return 1

    if hyp_len >= 25:
        fivegrams = [hyp_norm[i:i+5] for i in range(hyp_len - 4)]
        if Counter(fivegrams).most_common(1)[0][1] >= 6: return 1

    return 0

def normalize_gujarati(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize("NFKC", text)
    text = ZW_RE.sub("", text)
    text = text.translate(DIGITS_MAP)
    if _gu_indic_norm is not None:
        text = _gu_indic_norm.normalize(text)
    text = PUNCT_RE.sub(" ", text).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()

def compute_utterance_metrics(reference_raw: str, hypothesis_raw: str) -> dict:
    drift_flag, drift_ratio = detect_script_drift(hypothesis_raw)
    ref_norm = normalize_gujarati(reference_raw)
    hyp_norm = normalize_gujarati(hypothesis_raw)
    loop_flag = detect_loop_or_empty(ref_norm, hyp_norm)

    if not ref_norm:
        return {"ref_norm": "", "hyp_norm": hyp_norm, "wer": 0.0, "cer": 0.0, "capped_cer": 0.0, "script_drift": drift_flag, "drift_ratio": drift_ratio, "loop_flag": loop_flag}

    if not hyp_norm:
        raw_wer, raw_cer = 1.0, 1.0
    else:
        raw_wer = wer(ref_norm, hyp_norm)
        raw_cer = cer(ref_norm, hyp_norm)

    return {
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "wer": round(raw_wer * 100.0, 2),
        "cer": round(raw_cer * 100.0, 2),
        "capped_cer": round(min(1.0, raw_cer) * 100.0, 2),
        "script_drift": drift_flag,
        "drift_ratio": drift_ratio,
        "loop_flag": loop_flag,
    }

if __name__ == "__main__":
    test_pairs = [
        ("Anusvara", "અંદર", "અન્દર"),
        ("Devanagari drift", "તમે કેમ છો", "तमे केम छो"),
        ("Repetition loop", "આજે હવામાન સારું છે", "આજે હવામાન આજે હવામાન આજે હવામાન આજે હવામાન"),
        ("Empty output", "ગુજરાત", ""),
    ]
    for label, ref, hyp in test_pairs:
        m = compute_utterance_metrics(ref, hyp)
        print(f"{label:<18} | WER: {m['wer']:>5.1f}% | CER: {m['cer']:>5.1f}% | Drift: {m['script_drift']} | Loop: {m['loop_flag']}")
