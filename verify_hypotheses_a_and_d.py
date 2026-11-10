#!/usr/bin/env python3
"""empirical verification of hypothesis a (lexicon compound splitter/joiner)
and hypothesis d (short-clip ctc routing) on results/predictions_optimized.sqlite."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from jiwer import cer as corpus_cer, wer as corpus_wer
from tabulate import tabulate

from gu_normalizer import normalize_gujarati
from run_baselines_optimized import DATA_CACHE, DEFAULT_DB_PATH, hf_auth_token


def build_train_lexicon() -> tuple[Counter, Counter]:
    """builds word unigram and bigram counts strictly from train splits (zero test leakage)."""
    token = hf_auth_token()
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()

    print("[Lexicon] Building zero-leakage Gujarati word & bigram dictionary from train splits...")
    fleurs_pq = hf_hub_download(
        repo_id="google/fleurs",
        repo_type="dataset",
        filename="gu_in/train-00000-of-00001.parquet",
        revision="fdd74f7cdf01fc7641ab3072b6a38d364b3667f6",
        cache_dir=str(DATA_CACHE),
        token=token,
    )
    ds_fl = load_dataset("parquet", data_files={"train": fleurs_pq}, split="train", cache_dir=str(DATA_CACHE))
    ds_kb = load_dataset("ai4bharat/Kathbath", "gujarati", split="train", cache_dir=str(DATA_CACHE), token=token)
    ds_iv = load_dataset("ai4bharat/IndicVoices", "gujarati", split="train", cache_dir=str(DATA_CACHE), token=token)

    for text_list in (ds_fl["transcription"], ds_kb["text"], ds_iv["text"][:50000]):
        for raw in text_list:
            if not raw:
                continue
            words = normalize_gujarati(str(raw)).split()
            unigrams.update(words)
            for i in range(len(words) - 1):
                bigrams[(words[i], words[i + 1])] += 1

    print(f"[Lexicon] Indexed {len(unigrams):,} unique Gujarati words and {len(bigrams):,} bigrams.")
    return unigrams, bigrams


def apply_compound_splitter_joiner(text: str, unigrams: Counter, bigrams: Counter) -> str:
    """fixes space-boundary merges (ઓમપ્રકાશ -> ઓમ પ્રકાશ) and splits using train bigram statistics."""
    words = (text or "").split()
    if not words:
        return ""

    # Pass 1: Check adjacent words (w1, w2) to see if train corpus strongly prefers joined w1+w2
    merged_words: list[str] = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            w1, w2 = words[i], words[i + 1]
            joined = w1 + w2
            joined_cnt = unigrams[joined]
            sep_cnt = bigrams[(w1, w2)]
            if joined_cnt >= 15 and joined_cnt > sep_cnt * 3:
                merged_words.append(joined)
                i += 2
                continue
        merged_words.append(words[i])
        i += 1

    # Pass 2: Check out-of-vocabulary (or rare) long words to see if splitting into (p1, p2) is a frequent bigram
    final_words: list[str] = []
    for w in merged_words:
        if len(w) >= 6 and unigrams[w] <= 2:
            best_split = None
            best_score = 0
            for cut in range(2, len(w) - 1):
                p1, p2 = w[:cut], w[cut:]
                bg_cnt = bigrams[(p1, p2)]
                if bg_cnt >= 8 and unigrams[p1] >= 10 and unigrams[p2] >= 10:
                    if bg_cnt > best_score:
                        best_score = bg_cnt
                        best_split = (p1, p2)
            if best_split is not None:
                final_words.extend(best_split)
                continue
        final_words.append(w)

    return " ".join(final_words)


def main():
    unigrams, bigrams = build_train_lexicon()
    conn = sqlite3.connect(str(DEFAULT_DB_PATH))

    table = []
    for ds in ("fleurs", "indicvoices", "kathbath"):
        rows = conn.execute("""
            SELECT g.sample_id, g.audio_duration_s, g.reference_norm,
                   g.hypothesis_norm AS gemma_hyp, s.hypothesis_norm AS sravaani_hyp
            FROM predictions g
            LEFT JOIN predictions s
              ON g.dataset = s.dataset AND g.sample_id = s.sample_id AND s.model_name = 'sravaani-1.0'
            WHERE g.dataset = ? AND g.model_name = 'gemma-4-E4B-sft'
            ORDER BY g.row_index
        """, (ds,)).fetchall()

        refs = [r[2] for r in rows if r[2]]
        base_hyps = [r[3] or "" for r in rows if r[2]]
        hyp_a_hyps = [apply_compound_splitter_joiner(r[3] or "", unigrams, bigrams) for r in rows if r[2]]
        hyp_d_hyps = [
            (r[4] if (r[1] or 10.0) < 1.2 and r[4] else (r[3] or ""))
            for r in rows if r[2]
        ]

        base_cer, base_wer = 100.0 * corpus_cer(refs, base_hyps), 100.0 * corpus_wer(refs, base_hyps)
        a_cer, a_wer = 100.0 * corpus_cer(refs, hyp_a_hyps), 100.0 * corpus_wer(refs, hyp_a_hyps)
        d_cer, d_wer = 100.0 * corpus_cer(refs, hyp_d_hyps), 100.0 * corpus_wer(refs, hyp_d_hyps)

        table.append([
            ds, len(refs),
            f"{base_cer:.2f}% / {base_wer:.2f}%",
            f"{a_cer:.2f}% ({a_cer-base_cer:+.2f}%) / {a_wer:.2f}% ({a_wer-base_wer:+.2f}%)",
            f"{d_cer:.2f}% ({d_cer-base_cer:+.2f}%) / {d_wer:.2f}% ({d_wer-base_wer:+.2f}%)",
        ])

    conn.close()
    print("\n" + "=" * 115)
    print("EMPIRICAL VERIFICATION OF HYPOTHESIS A (LEXICON SPLITTER) & HYPOTHESIS D (<1.2s CTC ROUTER)")
    print("=" * 115)
    print(tabulate(
        table,
        headers=["Dataset", "N", "Step 6 SFT (CER / WER)", "Hypothesis A: Lexicon Post-Proc (CER / WER)", "Hypothesis D: <1.2s SraVaani Router (CER / WER)"],
        tablefmt="github",
    ))


if __name__ == "__main__":
    main()
