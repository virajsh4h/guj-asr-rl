#!/usr/bin/env python3
"""gujarati text normalizer and utterance-level asr metric calculator."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer

# map gujarati digits (૦-૯) to ascii digits (0-9) or vice versa consistently
GUJARATI_DIGITS = str.maketrans("૦૧૨૩૪૫૬૭૮૯", "0123456789")

# zero-width joiners and invisible formatting characters
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")

# punctuation to strip (ascii + indic danda/double-danda + smart quotes)
PUNCT_RE = re.compile(r"[^\w\s\u0A80-\u0AFF]", flags=re.UNICODE)
UNDERSCORE_RE = re.compile(r"_+")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_gujarati(text: str | None) -> str:
    """normalize gujarati transcript for fair wer/cer comparison."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = ZERO_WIDTH_RE.sub("", s)
    s = s.translate(GUJARATI_DIGITS)
    s = PUNCT_RE.sub(" ", s)
    s = UNDERSCORE_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip().lower()
    return s


def _compute_script_drift(raw_text: str) -> tuple[int, float]:
    """detect non-gujarati alphabetic script drift (e.g., latin or devanagari)."""
    if not raw_text:
        return 0, 0.0
    alpha_chars = [
        ch for ch in raw_text
        if ch.isalpha() or ("\u0900" <= ch <= "\u0DFF")
    ]
    if not alpha_chars:
        return 0, 0.0
    non_guj = sum(1 for ch in alpha_chars if not ("\u0A80" <= ch <= "\u0AFF"))
    ratio = non_guj / len(alpha_chars)
    # flag script drift if >10% of alphabetic chars or >=2 chars are non-gujarati
    drift_flag = 1 if (non_guj >= 2 and ratio > 0.05) or ratio > 0.15 else 0
    return drift_flag, round(ratio, 4)


def _detect_loop(norm_text: str) -> int:
    """flag repetition loops (4+ identical consecutive words or repeated n-grams)."""
    words = norm_text.split()
    if len(words) < 6:
        return 0
    # check consecutive single-word repeats (e.g. "અને અને અને અને")
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            if run >= 4:
                return 1
        else:
            run = 1
    # check repeated 2-gram or 3-gram phrases
    for n in (2, 3):
        if len(words) >= n * 4:
            ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            counts: dict[tuple[str, ...], int] = {}
            for ng in ngrams:
                counts[ng] = counts.get(ng, 0) + 1
                if counts[ng] >= 5:
                    return 1
    return 0


def compute_utterance_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    """compute normalized text, wer, cer, capped_cer, drift, and loop flags."""
    ref_norm = normalize_gujarati(reference)
    hyp_norm = normalize_gujarati(hypothesis)

    if not ref_norm:
        wer_val = 0.0 if not hyp_norm else 100.0
        cer_val = 0.0 if not hyp_norm else 100.0
    elif not hyp_norm:
        wer_val = 100.0
        cer_val = 100.0
    else:
        wer_val = 100.0 * float(jiwer_wer(ref_norm, hyp_norm))
        cer_val = 100.0 * float(jiwer_cer(ref_norm, hyp_norm))

    capped_cer = min(100.0, cer_val)
    script_drift, drift_ratio = _compute_script_drift(hypothesis or "")
    loop_flag = _detect_loop(hyp_norm)

    return {
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "wer": round(wer_val, 4),
        "cer": round(cer_val, 4),
        "capped_cer": round(capped_cer, 4),
        "script_drift": int(script_drift),
        "drift_ratio": float(drift_ratio),
        "loop_flag": int(loop_flag),
    }


# alias for backwards compatibility
compute_asr_metrics = compute_utterance_metrics
