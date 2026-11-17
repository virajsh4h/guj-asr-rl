import numpy as np

from gujasr import runner, store
from gujasr.adapters import ScriptedAdapter
from gujasr.config import Config


def _items(n=6):
    refs = ["એક બે ત્રણ", "ચાર પાંચ", "છ સાત", "આઠ નવ", "દસ", "અગિયાર"]
    return [
        {
            "dataset": "fleurs",
            "split": "test",
            "row_index": i,
            "sample_id": f"fleurs_{i:06d}",
            "audio_array": np.zeros(1600, dtype=np.float32),
            "duration_s": 4.0,
            "reference": refs[i % len(refs)],
        }
        for i in range(n)
    ]


class FakeDataset:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


def _loader(dataset_key, n=6):
    return (FakeDataset(n), "test", "audio", "transcription")


def _iterator_factory(items):
    def iterator(ds, dataset_key, split, audio_col, text_col, max_samples, done, workers, prefetch):
        limit = len(items) if max_samples <= 0 else min(len(items), max_samples)
        for item in items[:limit]:
            if item["sample_id"] not in done:
                yield item

    return iterator


def _cfg(**kwargs):
    base = dict(models=["whisper"], datasets=["fleurs"], db_path=":memory:", batch_size=2, max_samples=0)
    base.update(kwargs)
    return Config(**base)


def test_every_utterance_is_scored_and_stored():
    conn = store.connect(":memory:")
    adapter = ScriptedAdapter()
    summary = runner.evaluate(
        conn, _cfg(), adapter, "fleurs", "whisper", _loader, _iterator_factory(_items())
    )
    assert summary["rows"] == 6
    assert summary["expected"] == 6
    assert summary["coverage"] == 1.0
    assert len(store.fetch_predictions(conn, [summary["run_key"]])) == 6


def test_resume_skips_completed_utterances():
    conn = store.connect(":memory:")
    cfg = _cfg()
    first = runner.evaluate(
        conn, cfg, ScriptedAdapter(), "fleurs", "whisper", _loader, _iterator_factory(_items(4))
    )
    second = runner.evaluate(
        conn, cfg, ScriptedAdapter(), "fleurs", "whisper", _loader, _iterator_factory(_items(6))
    )
    assert first["run_key"] == second["run_key"]
    assert second["rows"] == 2
    assert len(store.fetch_predictions(conn, [first["run_key"]])) == 6


def test_changing_the_config_changes_the_run_identity():
    a = _cfg(batch_size=2).run_key("whisper", "fleurs")
    b = _cfg(batch_size=8).run_key("whisper", "fleurs")
    assert a != b


def test_failures_are_recorded_and_excluded():
    conn = store.connect(":memory:")
    summary = runner.evaluate(
        conn, _cfg(), ScriptedAdapter(fail_on={3}), "fleurs", "whisper", _loader, _iterator_factory(_items())
    )
    assert summary["failures"] == 1
    keep = store.fetch_predictions(conn, [summary["run_key"]])
    assert len(keep) == 5
    assert all(row["sample_id"] != "fleurs_000003" for row in keep)


def test_max_samples_limits_coverage_honestly():
    conn = store.connect(":memory:")
    summary = runner.evaluate(
        conn, _cfg(max_samples=3), ScriptedAdapter(), "fleurs", "whisper", _loader, _iterator_factory(_items())
    )
    assert summary["rows"] == 3
    assert summary["expected"] == 3
