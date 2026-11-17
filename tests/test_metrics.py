import pytest

from gujasr import metrics

REFS = ["આજે હવામાન સારું છે", "હું ઘરે જાઉં છું", "ગુજરાતી ભાષા સુંદર છે"]
GOOD = ["આજે હવામાન સારું છે", "હું ઘરે જાઉં છું", "ગુજરાતી ભાષા સુંદર છે"]
MIXED = ["આજે હવામાન સારું છે", "હું ઘરે છું", "ગુજરાતી ભાષા છે"]
BAD = ["", "", ""]


def test_identical_output_scores_zero():
    assert metrics.corpus_rate(REFS, GOOD) == 0.0


def test_corpus_rate_is_edit_weighted_not_mean_of_sentences():
    refs = ["એક બે ત્રણ ચાર", "એક"]
    data = metrics.counts(refs, ["એક બે ત્રણ ચાર", "બે"])
    per_sentence = metrics.macro([0.0, 100.0])
    assert data.rate == pytest.approx(100.0 * data.edits.sum() / data.lengths.sum())
    assert data.rate < per_sentence


def test_word_rate_uses_word_units():
    data = metrics.counts(["એક બે ત્રણ"], ["એક બે"], "word")
    assert data.rate == pytest.approx(100.0 * 1 / 3)


def test_empty_reference_is_skipped():
    data = metrics.counts(["", "એક બે"], ["કંઈક", "એક બે"])
    assert data.n == 1
    assert data.rate == 0.0


def test_bootstrap_interval_brackets_the_point_estimate():
    data = metrics.counts(REFS + MIXED, MIXED + MIXED)
    lo, hi = metrics.bootstrap_ci(data, reps=400, seed=1)
    assert lo <= data.rate <= hi
    assert hi - lo > 0


def test_identical_models_are_not_significantly_different():
    left = metrics.counts(REFS, MIXED)
    right = metrics.counts(REFS, MIXED)
    result = metrics.paired_bootstrap(left, right, reps=400, seed=2)
    assert result.delta == 0.0
    assert result.p_value > 0.05


def test_a_clearly_worse_model_is_significant():
    refs = ["એક બે ત્રણ ચાર પાંચ છ સાત આઠ નવ દસ"] * 40
    left = metrics.counts(refs, ["" for _ in refs])
    right = metrics.counts(refs, refs)
    result = metrics.paired_bootstrap(left, right, reps=400, seed=3)
    assert result.delta == pytest.approx(100.0)
    assert result.p_value < 0.05
    assert result.significant


def test_paired_requires_equal_length():
    with pytest.raises(ValueError):
        metrics.paired_bootstrap(
            metrics.counts(REFS, GOOD), metrics.counts(REFS[:2], GOOD[:2])
        )


def test_macro_is_the_plain_mean():
    assert metrics.macro([3.0, 5.0]) == pytest.approx(4.0)
    assert metrics.macro([]) == 0.0
