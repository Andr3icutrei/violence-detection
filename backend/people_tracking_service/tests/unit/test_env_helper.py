import os

import pytest

from helpers import env_helper


def test_get_env_variable_returns_value(monkeypatch):
    env_helper.load_env_once.cache_clear()
    monkeypatch.setenv("TEST_ENV_VALUE", "present")

    assert env_helper.get_env_variable("TEST_ENV_VALUE") == "present"


def test_get_env_variable_returns_default(monkeypatch):
    env_helper.load_env_once.cache_clear()
    monkeypatch.delenv("TEST_ENV_MISSING", raising=False)

    assert env_helper.get_env_variable("TEST_ENV_MISSING", "fallback") == "fallback"


def test_get_env_variable_raises_when_missing(monkeypatch):
    env_helper.load_env_once.cache_clear()
    monkeypatch.delenv("TEST_ENV_REQUIRED", raising=False)

    with pytest.raises(ValueError):
        env_helper.get_env_variable("TEST_ENV_REQUIRED")

