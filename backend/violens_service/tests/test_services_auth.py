import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from services import auth_service
from services.auth_service import AuthService, _get_user_from_token


def run(coro):
    return asyncio.run(coro)


def make_user(**overrides):
    data = {
        "id": 1,
        "email": "user@example.com",
        "is_account_verified": True,
        "is_banned": False,
        "hashed_password": "hashed",
        "auth_provider": "local",
        "is_admin": False,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_login_user_not_found():
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=None)
    service = AuthService(repo)

    with pytest.raises(HTTPException) as exc:
        run(service.login("missing@example.com", "pass"))

    assert exc.value.status_code == 404


def test_login_not_verified():
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=make_user(is_account_verified=False))
    service = AuthService(repo)

    with pytest.raises(HTTPException) as exc:
        run(service.login("user@example.com", "pass"))

    assert exc.value.status_code == 403


def test_login_banned():
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=make_user(is_banned=True))
    service = AuthService(repo)

    with pytest.raises(HTTPException) as exc:
        run(service.login("user@example.com", "pass"))

    assert exc.value.status_code == 403


def test_login_wrong_password(monkeypatch):
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=make_user())
    monkeypatch.setattr(auth_service, "verify_password", lambda *_: False)
    service = AuthService(repo)

    with pytest.raises(HTTPException) as exc:
        run(service.login("user@example.com", "wrong"))

    assert exc.value.status_code == 401


def test_login_success(monkeypatch):
    user = make_user()
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=user)
    monkeypatch.setattr(auth_service, "verify_password", lambda *_: True)
    service = AuthService(repo)

    result = run(service.login("user@example.com", "ok"))

    assert result == user


def test_login_google_creates_user(monkeypatch):
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=None)
    repo.create_user_google = AsyncMock(return_value=make_user(email="new@example.com", auth_provider="google"))
    monkeypatch.setattr(auth_service, "get_env_variable", lambda *_: "client-id")
    monkeypatch.setattr(
        auth_service.id_token,
        "verify_oauth2_token",
        lambda *_args, **_kwargs: {"email": "new@example.com"},
    )
    service = AuthService(repo)

    result = run(service.login_google(SimpleNamespace(tokenId="token")))

    assert result.email == "new@example.com"
    repo.create_user_google.assert_awaited_once()


def test_login_google_banned_user(monkeypatch):
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=make_user(is_banned=True))
    monkeypatch.setattr(auth_service, "get_env_variable", lambda *_: "client-id")
    monkeypatch.setattr(
        auth_service.id_token,
        "verify_oauth2_token",
        lambda *_args, **_kwargs: {"email": "user@example.com"},
    )
    service = AuthService(repo)

    with pytest.raises(HTTPException) as exc:
        run(service.login_google(SimpleNamespace(tokenId="token")))

    assert exc.value.status_code == 403


def test_login_google_updates_provider(monkeypatch):
    user = make_user(auth_provider="local")
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=user)
    repo.save_changes = AsyncMock(return_value=None)
    monkeypatch.setattr(auth_service, "get_env_variable", lambda *_: "client-id")
    monkeypatch.setattr(
        auth_service.id_token,
        "verify_oauth2_token",
        lambda *_args, **_kwargs: {"email": "user@example.com"},
    )
    service = AuthService(repo)

    result = run(service.login_google(SimpleNamespace(tokenId="token")))

    assert result.auth_provider == "google"
    repo.save_changes.assert_awaited_once_with(user)


def test_get_user_from_token_expired(monkeypatch):
    repo = Mock()
    repo.get_by_id = AsyncMock()
    monkeypatch.setattr(auth_service, "decode_jwt_token", lambda *_: None)

    with pytest.raises(HTTPException) as exc:
        run(_get_user_from_token("token", repo))

    assert exc.value.status_code == 401


def test_get_user_from_token_user_not_found(monkeypatch):
    repo = Mock()
    repo.get_by_id = AsyncMock(return_value=None)
    monkeypatch.setattr(auth_service, "decode_jwt_token", lambda *_: {"sub": "1"})

    with pytest.raises(HTTPException) as exc:
        run(_get_user_from_token("token", repo))

    assert exc.value.status_code == 404


def test_get_user_from_token_success(monkeypatch):
    user = make_user()
    repo = Mock()
    repo.get_by_id = AsyncMock(return_value=user)
    monkeypatch.setattr(auth_service, "decode_jwt_token", lambda *_: {"sub": "1"})

    result = run(_get_user_from_token("token", repo))

    assert result == user
