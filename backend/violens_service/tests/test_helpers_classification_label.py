from helpers.classification_label_helper import is_violent_label


def test_is_violent_label_positive_default():
    assert is_violent_label("Violent fight") is True


def test_is_violent_label_negative_default():
    assert is_violent_label("Non-violent dispute") is False


def test_is_violent_label_env_override(monkeypatch):
    monkeypatch.setenv("CLASSIFICATION_POSITIVE_LABELS", "hit,attack")
    monkeypatch.setenv("CLASSIFICATION_NEGATIVE_LABELS", "safe,calm")
    assert is_violent_label("attack scene") is True
    assert is_violent_label("calm attack") is False


def test_is_violent_label_unknown_returns_false():
    assert is_violent_label("friendly chat") is False

