from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.dependencies.token_from_cookie import get_token_from_cookie


def test_get_token_from_cookie_success():
    request = SimpleNamespace(cookies={"access_token": "token"})

    assert get_token_from_cookie(request) == "token"


def test_get_token_from_cookie_missing():
    request = SimpleNamespace(cookies={})

    with pytest.raises(HTTPException) as exc:
        get_token_from_cookie(request)

    assert exc.value.status_code == 401

