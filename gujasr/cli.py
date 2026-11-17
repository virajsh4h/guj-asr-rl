from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from . import manifests, report, store
from .config import Config


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_run(args) -> int:
    from . import runner

    cfg = Config.load(args.config) if args.config else Config()
    if args.models:
        cfg.models = args.models
    if args.datasets:
        cfg.datasets = args.datasets
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.db:
        cfg.db_path = args.db
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if not cfg.models or not cfg.datasets:
        print("nothing to run: pass --models and --datasets", file=sys.stderr)
        return 2
    summaries = runner.run(cfg)
    _print(summaries)
    return 0 if all(s["rows"] for s in summaries) else 1


def cmd_report(args) -> int:
    cfg = Config.load(args.config) if args.config else Config()
    if args.db:
        cfg.db_path = args.db
    conn = store.connect(cfg.db_path)
    try:
        board = report.build(
            conn,
            cfg,
            run_keys=args.run_keys or None,
            reference=args.reference,
        )
        paths = report.write(board, Path(args.out))
    finally:
        conn.close()
    _print(paths)
    return 0


def cmd_import(args) -> int:
    source = sqlite3.connect(args.source)
    source.row_factory = sqlite3.Row
    by_name = {spec["name"]: key for key, spec in manifests.MODELS.items()}
    rows = source.execute("SELECT * FROM predictions").fetchall()
    conn = store.connect(args.db)
    written = 0
    skipped = 0
    seen: dict[tuple[str, str, str], int] = {}
    try:
        payload = []
        for row in rows:
            model_key = by_name.get(row["model_name"])
            if model_key is None:
                skipped += 1
                continue
            run_key = f"{args.run_key}:{model_key}"
            seen[(model_key, row["dataset"], row["split"])] = seen.get(
                (model_key, row["dataset"], row["split"]), 0
            ) + 1
            payload.append(
                {
                    "run_key": run_key,
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "sample_id": row["sample_id"],
                    "row_index": row["row_index"],
                    "model_key": model_key,
                    "model_name": row["model_name"],
                    "audio_duration_s": row["audio_duration_s"],
                    "latency_s": row["latency_s"],
                    "rtf": row["rtf"],
                    "reference_raw": row["reference_raw"],
                    "hypothesis_raw": row["hypothesis_raw"] or "",
                    "reference_norm": row["reference_norm"],
                    "hypothesis_norm": row["hypothesis_norm"],
                    "wer": row["wer"],
                    "cer": row["cer"],
                    "capped_cer": row["capped_cer"],
                    "script_drift": row["script_drift"],
                    "drift_ratio": row["drift_ratio"],
                    "loop_flag": row["loop_flag"],
                    "failure_tag": row["failure_tag"] or "",
                }
            )
            if len(payload) >= 5000:
                written += store.write_rows(conn, payload)
                payload = []
        written += store.write_rows(conn, payload)
        for (model_key, dataset, split), count in seen.items():
            conn.execute(
                """INSERT OR REPLACE INTO runs (run_key, created_at, model_key, model_name,
                       dataset, split, config_hash, code_sha, normalizer_version,
                       manifest_json, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"{args.run_key}:{model_key}", args.created_at, model_key,
                    manifests.model_name(model_key), dataset, split,
                    "import", "unknown", "gu-normalizer/1",
                    json.dumps(manifests.dataset_manifest(dataset)), "imported",
                ),
            )
        conn.commit()
    finally:
        conn.close()
        source.close()
    _print({"run_key": args.run_key, "written": written, "skipped": skipped})
    return 0


def cmd_verify(args) -> int:
    from .normalize import utterance_metrics

    conn = store.connect(args.db)
    mismatches = []
    checked = 0
    try:
        for row in store.fetch_predictions(conn, args.run_keys or None):
            metrics = utterance_metrics(row["reference_raw"], row["hypothesis_raw"])
            checked += 1
            norm_differs = metrics["ref_norm"] != row["reference_norm"]
            if norm_differs or abs(metrics["cer"] - row["cer"]) > 0.05:
                mismatches.append(
                    {"sample_id": row["sample_id"], "model": row["model_name"],
                     "norm_differs": norm_differs,
                     "stored": row["cer"], "recomputed": metrics["cer"]}
                )
    finally:
        conn.close()
    _print({"checked": checked, "mismatches": len(mismatches), "examples": mismatches[:5]})
    return 0 if not mismatches else 1


def cmd_models(args) -> int:
    _print({key: spec["name"] for key, spec in manifests.MODELS.items()})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gujasr", description="Gujarati ASR benchmark harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="evaluate models and write predictions")
    p.add_argument("--config")
    p.add_argument("--models", nargs="+")
    p.add_argument("--datasets", nargs="+", choices=list(manifests.DATASETS))
    p.add_argument("--max-samples", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--db")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="build leaderboard artefacts")
    p.add_argument("--config")
    p.add_argument("--db")
    p.add_argument("--out", default="results/harness")
    p.add_argument("--run-keys", nargs="*")
    p.add_argument("--reference")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("import", help="ingest predictions from the original bench database")
    p.add_argument("--source", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--run-key", required=True)
    p.add_argument("--created-at", default="2026-11-16T00:00:00+00:00")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("verify", help="recompute metrics from stored raw text")
    p.add_argument("--db", required=True)
    p.add_argument("--run-keys", nargs="*")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("models", help="list model keys")
    p.set_defaults(func=cmd_models)

    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
