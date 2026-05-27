import pytest

from helpers.env_helper import get_env_float, get_env_variable


def test_get_env_variable_returns_default_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    assert get_env_variable("MISSING_KEY", "value") == "value"


def test_get_env_variable_raises_when_missing_and_no_default(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(ValueError):
        get_env_variable("MISSING_KEY")


def test_get_env_float_parses_value(monkeypatch):
    monkeypatch.setenv("FLOAT_KEY", "1.25")
    assert get_env_float("FLOAT_KEY") == 1.25


def test_get_env_float_raises_on_invalid(monkeypatch):
    monkeypatch.setenv("FLOAT_KEY", "bad")
    with pytest.raises(ValueError):
        get_env_float("FLOAT_KEY")

