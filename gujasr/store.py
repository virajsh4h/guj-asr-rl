"""SQLite store keyed by run identity.

The old bench resumed by sample id alone, so a row produced under one normaliser or one
decoding config was silently reusable by another. Here a row belongs to a `run_key` that
hashes the model, dataset, scoring config, code revision and normaliser version. Resume
can only ever see its own run.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_key TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    code_sha TEXT,
    normalizer_version TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS predictions (
    run_key TEXT NOT NULL,
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    model_key TEXT NOT NULL,
    model_name TEXT NOT NULL,
    audio_duration_s REAL NOT NULL,
    latency_s REAL,
    rtf REAL,
    reference_raw TEXT NOT NULL,
    hypothesis_raw TEXT NOT NULL DEFAULT '',
    reference_norm TEXT NOT NULL,
    hypothesis_norm TEXT NOT NULL,
    wer REAL NOT NULL,
    cer REAL NOT NULL,
    capped_cer REAL NOT NULL,
    script_drift INTEGER NOT NULL DEFAULT 0,
    drift_ratio REAL NOT NULL DEFAULT 0,
    loop_flag INTEGER NOT NULL DEFAULT 0,
    failure_tag TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_key, dataset, sample_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions (run_key, dataset);
"""

COLUMNS = (
    "run_key", "dataset", "split", "sample_id", "row_index", "model_key", "model_name",
    "audio_duration_s", "latency_s", "rtf", "reference_raw", "hypothesis_raw",
    "reference_norm", "hypothesis_norm", "wer", "cer", "capped_cer", "script_drift",
    "drift_ratio", "loop_flag", "failure_tag",
)


def connect(path: str | Path) -> sqlite3.Connection:
    if str(path) == ":memory:":
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        return conn
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def start_run(conn: sqlite3.Connection, run: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO runs (run_key, created_at, model_key, model_name, dataset, split,
                             config_hash, code_sha, normalizer_version, manifest_json, status)
           VALUES (:run_key, :created_at, :model_key, :model_name, :dataset, :split,
                   :config_hash, :code_sha, :normalizer_version, :manifest_json, :status)
           ON CONFLICT(run_key) DO UPDATE SET status=excluded.status""",
        run,
    )
    conn.commit()


def set_status(conn: sqlite3.Connection, run_key: str, status: str) -> None:
    conn.execute("UPDATE runs SET status=? WHERE run_key=?", (status, run_key))
    conn.commit()


def completed_ids(conn: sqlite3.Connection, run_key: str) -> set[str]:
    rows = conn.execute("SELECT sample_id FROM predictions WHERE run_key=?", (run_key,)).fetchall()
    return {row["sample_id"] for row in rows}


def write_rows(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    payload = [tuple(row.get(col) for col in COLUMNS) for row in rows]
    if not payload:
        return 0
    conn.executemany(
        f"INSERT OR REPLACE INTO predictions ({', '.join(COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in COLUMNS)})",
        payload,
    )
    conn.commit()
    return len(payload)


def fetch_predictions(
    conn: sqlite3.Connection,
    run_keys: Iterable[str] | None = None,
    exclude_failures: bool = True,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM predictions"
    args: list[Any] = []
    clauses = []
    if run_keys is not None:
        keys = list(run_keys)
        if not keys:
            return []
        clauses.append("run_key IN (%s)" % ",".join("?" for _ in keys))
        args.extend(keys)
    if exclude_failures:
        clauses.append("failure_tag = ''")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY dataset, model_key, row_index"
    return [dict(row) for row in conn.execute(sql, args).fetchall()]


def list_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute("SELECT * FROM runs ORDER BY created_at").fetchall()]


def dump_manifest(conn: sqlite3.Connection, run_key: str) -> dict[str, Any]:
    row = conn.execute("SELECT manifest_json FROM runs WHERE run_key=?", (run_key,)).fetchone()
    return json.loads(row["manifest_json"]) if row else {}
