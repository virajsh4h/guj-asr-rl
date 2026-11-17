from gujasr import normalize


def test_gujarati_digits_become_ascii():
    assert normalize.normalize("૨૦૨૬") == "2026"


def test_zero_width_and_punctuation_are_removed():
    assert normalize.normalize("ગુજરાતી, ભાષા!") == "ગુજરાતી ભાષા"


def test_empty_input_is_empty():
    assert normalize.normalize("") == ""
    assert normalize.normalize(None) == ""


def test_same_text_normalises_identically():
    a = normalize.utterance_metrics("આજે હવામાન સારું છે", "આજે હવામાન સારું છે.")
    assert a["cer"] == 0.0
    assert a["wer"] == 0.0


def test_script_drift_flags_latin_output():
    out = normalize.utterance_metrics("આજે હવામાન સારું છે", "aaje havaaman saaru chhe")
    assert out["script_drift"] == 1


def test_loop_flag_catches_repetition():
    out = normalize.utterance_metrics("એક બે ત્રણ", "એક એક એક એક એક એક એક")
    assert out["loop_flag"] == 1


def test_capped_cer_never_exceeds_one_hundred():
    out = normalize.utterance_metrics("એક", "એક " * 200)
    assert out["cer"] > 100.0
    assert out["capped_cer"] == 100.0
