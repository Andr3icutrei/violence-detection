import types
import pytest

from services.inference_runtime import _resolve_symbol


def test_resolve_symbol_finds_first_match():
    module = types.SimpleNamespace(alpha=1, beta=2)

    assert _resolve_symbol(module, "beta", "alpha") == 2


def test_resolve_symbol_raises_when_missing():
    module = types.SimpleNamespace(alpha=1)

    with pytest.raises(AttributeError):
        _resolve_symbol(module, "missing", "other")

