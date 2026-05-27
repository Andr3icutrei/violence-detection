import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import jwt
import pytest
from fastapi import HTTPException

from services import users_service
from services.users_service import UsersService


def run(coro):
    return asyncio.run(coro)


def make_user(**overrides):
    data = {
        "id": 1,
        "email": "user@example.com",
        "is_admin": False,
        "credits": 10,
        "is_account_verified": False,
        "is_banned": False,
        "hashed_password": "hashed",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def make_service(repo=None, notifier=None):
    return UsersService(repo or Mock(), notifier or Mock())


def test_create_user_conflict():
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=make_user())
    service = make_service(repo=repo)

    with pytest.raises(HTTPException) as exc:
        run(service.create_user(SimpleNamespace(email="user@example.com", password="pass"), conf=Mock()))

    assert exc.value.status_code == 409


def test_create_user_hash_error(monkeypatch):
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=None)
    monkeypatch.setattr(users_service, "get_password_hash", lambda *_: (_ for _ in ()).throw(ValueError("bad")))
    service = make_service(repo=repo)

    with pytest.raises(HTTPException) as exc:
        run(service.create_user(SimpleNamespace(email="user@example.com", password="pass"), conf=Mock()))

    assert exc.value.status_code == 400


def test_create_user_sends_email(monkeypatch):
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=None)
    repo.create_user = AsyncMock(return_value=make_user(is_account_verified=False))
    monkeypatch.setattr(users_service, "get_password_hash", lambda *_: "hashed")
    monkeypatch.setattr(users_service, "create_jwt_token", lambda *_args, **_kwargs: "token")
    send_email = AsyncMock(return_value=None)
    monkeypatch.setattr(users_service, "send_registration_email", send_email)
    service = make_service(repo=repo)

    result = run(service.create_user(SimpleNamespace(email="user@example.com", password="pass"), conf=Mock()))

    assert result.email == "user@example.com"
    send_email.assert_awaited_once()


def test_verify_account_invalid_token(monkeypatch):
    repo = Mock()
    monkeypatch.setattr(users_service, "decode_jwt_token", lambda *_: None)
    service = make_service(repo=repo)

    with pytest.raises(HTTPException) as exc:
        run(service.verify_account("token"))

    assert exc.value.status_code == 401


def test_verify_account_email_mismatch(monkeypatch):
    repo = Mock()
    repo.get_by_id = AsyncMock(return_value=make_user(email="user@example.com"))
    monkeypatch.setattr(users_service, "decode_jwt_token", lambda *_: {"email": "other@example.com", "sub": "1"})
    service = make_service(repo=repo)

    with pytest.raises(HTTPException) as exc:
        run(service.verify_account("token"))

    assert exc.value.status_code == 400


def test_verify_account_success(monkeypatch):
    user = make_user(is_account_verified=False)
    repo = Mock()
    repo.get_by_id = AsyncMock(return_value=user)
    repo.save_changes = AsyncMock(return_value=None)
    monkeypatch.setattr(users_service, "decode_jwt_token", lambda *_: {"email": user.email, "sub": "1"})
    service = make_service(repo=repo)

    result = run(service.verify_account("token"))

    assert result.is_account_verified is True
    repo.save_changes.assert_awaited_once_with(user)


def test_resend_verification_email(monkeypatch):
    user = make_user(is_account_verified=False)
    repo = Mock()
    repo.get_by_id = AsyncMock(return_value=user)
    monkeypatch.setattr(users_service, "decode_jwt_token_without_exp_check", lambda *_: {"email": user.email, "sub": "1"})
    monkeypatch.setattr(users_service, "create_jwt_token", lambda *_args, **_kwargs: "token")
    send_email = AsyncMock(return_value=None)
    monkeypatch.setattr(users_service, "send_registration_email", send_email)
    service = make_service(repo=repo)

    result = run(service.resend_verification_email("token", conf=Mock()))

    assert result.email == user.email
    send_email.assert_awaited_once()


def test_send_reset_password_email_for_verified_user(monkeypatch):
    user = make_user(is_account_verified=True)
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=user)
    monkeypatch.setattr(users_service, "create_jwt_token", lambda *_args, **_kwargs: "token")
    send_email = AsyncMock(return_value=None)
    monkeypatch.setattr(users_service, "send_reset_password_email", send_email)
    service = make_service(repo=repo)

    result = run(service.send_reset_password_email(user.email, conf=Mock()))

    assert result.email == user.email
    send_email.assert_awaited_once()


def test_reset_password_invalid_token(monkeypatch):
    repo = Mock()
    monkeypatch.setattr(users_service.jwt, "decode", lambda *_args, **_kwargs: (_ for _ in ()).throw(jwt.PyJWTError()))
    service = make_service(repo=repo)

    with pytest.raises(HTTPException) as exc:
        run(service.reset_password("token", "new"))

    assert exc.value.status_code == 400


def test_reset_password_success(monkeypatch):
    user = make_user(is_account_verified=True)
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=user)
    repo.save_changes = AsyncMock(return_value=None)
    monkeypatch.setattr(users_service.jwt, "decode", lambda *_args, **_kwargs: {"email": user.email})
    monkeypatch.setattr(users_service, "decode_jwt_token_reset_password", lambda *_: {"sub": "1"})
    monkeypatch.setattr(users_service, "get_password_hash", lambda *_: "new-hash")
    service = make_service(repo=repo)

    result = run(service.reset_password("token", "new"))

    assert result.hashed_password == "new-hash"
    repo.save_changes.assert_awaited_once_with(user)


def test_verify_reset_password_token_invalid_format(monkeypatch):
    repo = Mock()
    monkeypatch.setattr(users_service.jwt, "decode", lambda *_args, **_kwargs: (_ for _ in ()).throw(jwt.PyJWTError()))
    service = make_service(repo=repo)

    with pytest.raises(HTTPException) as exc:
        run(service.verify_reset_password_token("token"))

    assert exc.value.status_code == 400


def test_verify_reset_password_token_success(monkeypatch):
    user = make_user(is_account_verified=True)
    repo = Mock()
    repo.get_by_email = AsyncMock(return_value=user)
    monkeypatch.setattr(users_service.jwt, "decode", lambda *_args, **_kwargs: {"email": user.email})
    monkeypatch.setattr(users_service, "decode_jwt_token_reset_password", lambda *_: {"sub": "1"})
    service = make_service(repo=repo)

    result = run(service.verify_reset_password_token("token"))

    assert result is True


def test_update_all_users_credits(monkeypatch):
    repo = Mock()
    repo.get_all_exclude_banned = AsyncMock(return_value=[make_user(), make_user(id=2)])
    repo.update_users_credits = AsyncMock(return_value=None)
    monkeypatch.setattr(users_service, "get_env_variable", lambda *_: "15")
    service = make_service(repo=repo)

    run(service.update_all_users_credits())

    repo.update_users_credits.assert_awaited_once()


def test_update_user_role_broadcasts(monkeypatch):
    user = make_user(is_admin=False)
    repo = Mock()
    repo.get_by_id = AsyncMock(return_value=user)
    repo.save_changes = AsyncMock(return_value=None)
    notifier = Mock()
    notifier.broadcast_user_updated = AsyncMock(return_value=None)
    service = make_service(repo=repo, notifier=notifier)

    run(service.update_user_role(1, True))

    assert user.is_admin is True
    notifier.broadcast_user_updated.assert_awaited_once_with(1)


def test_ban_user_sends_email(monkeypatch):
    user = make_user(is_banned=False)
    repo = Mock()
    repo.get_by_id = AsyncMock(return_value=user)
    repo.save_changes = AsyncMock(return_value=None)
    notifier = Mock()
    notifier.broadcast_user_updated = AsyncMock(return_value=None)
    send_email = AsyncMock(return_value=None)
    monkeypatch.setattr(users_service, "send_user_ban_email", send_email)
    service = make_service(repo=repo, notifier=notifier)

    run(service.ban_user(1, "reason", conf=Mock()))

    assert user.is_banned is True
    send_email.assert_awaited_once()
    notifier.broadcast_user_updated.assert_awaited_once_with(1)


def test_get_users_stats():
    repo = Mock()
    repo.count_active_users = AsyncMock(return_value=2)
    repo.count_banned_users = AsyncMock(return_value=1)
    repo.count_inactive_users = AsyncMock(return_value=3)
    user = make_hashable_user(id=5)
    repo.get_most_active_users = AsyncMock(return_value={user: 7})
    service = make_service(repo=repo)

    result = run(service.get_users_stats())

    assert result.active_users == 2
    assert result.banned_users == 1
    assert result.inactive_users == 3
    assert result.most_active_users[0].id == 5


def make_hashable_user(**overrides):
    user = type("UserObj", (), {})()
    user.id = 1
    user.email = "user@example.com"
    user.is_admin = False
    user.credits = 10
    user.is_account_verified = False
    user.is_banned = False
    user.hashed_password = "hashed"
    for key, value in overrides.items():
        setattr(user, key, value)
    return user
