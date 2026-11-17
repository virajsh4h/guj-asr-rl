from gujasr import store


def _row(run_key, sample_id="fleurs_000000", failure=""):
    return {
        "run_key": run_key,
        "dataset": "fleurs",
        "split": "test",
        "sample_id": sample_id,
        "row_index": int(sample_id.split("_")[-1]),
        "model_key": "whisper",
        "model_name": "whisper-small",
        "audio_duration_s": 4.0,
        "latency_s": 0.1,
        "rtf": 0.025,
        "reference_raw": "એક બે",
        "hypothesis_raw": "એક બે",
        "reference_norm": "એક બે",
        "hypothesis_norm": "એક બે",
        "wer": 0.0,
        "cer": 0.0,
        "capped_cer": 0.0,
        "script_drift": 0,
        "drift_ratio": 0.0,
        "loop_flag": 0,
        "failure_tag": failure,
    }


def _run(run_key, model_key="whisper"):
    return {
        "run_key": run_key,
        "created_at": "2026-11-17T00:00:00+00:00",
        "model_key": model_key,
        "model_name": "whisper-small",
        "dataset": "fleurs",
        "split": "test",
        "config_hash": "abc",
        "code_sha": "deadbeef",
        "normalizer_version": "gu-normalizer/1",
        "manifest_json": "{}",
        "status": "running",
    }


def test_resume_state_is_scoped_to_one_run_key():
    conn = store.connect(":memory:")
    store.start_run(conn, _run("run-a"))
    store.start_run(conn, _run("run-b"))
    store.write_rows(conn, [_row("run-a")])

    assert store.completed_ids(conn, "run-a") == {"fleurs_000000"}
    assert store.completed_ids(conn, "run-b") == set()


def test_failures_are_stored_but_excluded_from_scoring():
    conn = store.connect(":memory:")
    store.start_run(conn, _run("run-a"))
    store.write_rows(conn, [_row("run-a", "fleurs_000000"), _row("run-a", "fleurs_000001", "RuntimeError('x')")])

    assert len(store.fetch_predictions(conn, ["run-a"])) == 1
    assert len(store.fetch_predictions(conn, ["run-a"], exclude_failures=False)) == 2


def test_status_transitions_are_persisted():
    conn = store.connect(":memory:")
    store.start_run(conn, _run("run-a"))
    store.set_status(conn, "run-a", "complete")
    assert store.list_runs(conn)[0]["status"] == "complete"


def test_writes_are_idempotent():
    conn = store.connect(":memory:")
    store.start_run(conn, _run("run-a"))
    store.write_rows(conn, [_row("run-a")])
    store.write_rows(conn, [_row("run-a")])
    assert len(store.fetch_predictions(conn, ["run-a"])) == 1
