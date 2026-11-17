from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

CARD_HEADER = """---
license: cc-by-4.0
task_categories:
  - automatic-speech-recognition
language:
  - gu
pretty_name: Gujarati ASR leaderboard
tags:
  - asr
  - gujarati
  - benchmark
  - evaluation
---
"""


def export_predictions(db_path: str, out_dir: Path) -> int:
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    models = [r["model_key"] for r in conn.execute(
        "SELECT DISTINCT model_key FROM predictions ORDER BY model_key"
    ).fetchall()]
    total = 0
    for model_key in models:
        rows = conn.execute(
            """SELECT dataset, split, sample_id, row_index, audio_duration_s, latency_s,
                      reference_raw, hypothesis_raw, reference_norm, hypothesis_norm,
                      wer, cer, capped_cer, script_drift, loop_flag, failure_tag
               FROM predictions WHERE model_key=? ORDER BY dataset, row_index""",
            (model_key,),
        ).fetchall()
        path = out_dir / "predictions" / f"{model_key}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(rows[0].keys() if rows else [])
            for row in rows:
                writer.writerow([row[k] for k in row.keys()])
        total += len(rows)
    conn.close()
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="publish leaderboard artefacts to HF")
    parser.add_argument("--repo", required=True, help="e.g. virajsh4h/gujarati-asr-leaderboard")
    parser.add_argument("--results", default="results/harness")
    parser.add_argument("--db", default="results/harness.sqlite")
    parser.add_argument("--card", default="docs/hf-card.md")
    parser.add_argument("--message", default="Update Gujarati ASR leaderboard")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    results = Path(args.results)
    staging = Path(tempfile.mkdtemp(prefix="hf-gujasr-"))
    try:
        for name in ("leaderboard.csv", "leaderboard.json", "leaderboard.md", "report.html"):
            source = results / name
            if source.exists():
                shutil.copy2(source, staging / name)
        count = export_predictions(args.db, staging)
        card = Path(args.card)
        if card.exists():
            (staging / "README.md").write_text(CARD_HEADER + card.read_text(encoding="utf-8"), encoding="utf-8")
        (staging / "manifest.json").write_text(
            json.dumps({"predictions": count, "source": args.db}, indent=2), encoding="utf-8"
        )

        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo,
            repo_type="dataset",
            folder_path=str(staging),
            commit_message=args.message,
        )
        print(json.dumps({"repo": args.repo, "private": not args.public, "predictions": count}))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
