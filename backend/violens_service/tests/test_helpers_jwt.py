from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException

from helpers.jwt_helper import (
    create_jwt_token,
    decode_jwt_token,
    decode_jwt_token_reset_password,
    decode_jwt_token_without_exp_check,
)


@pytest.fixture(autouse=True)
def jwt_env(monkeypatch):
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("SECRET_JWT_KEY", "secret")
    monkeypatch.setenv("SECRET_JWT_EMAIL", "email-secret")


def test_create_and_decode_jwt_token_success():
    token = create_jwt_token({"sub": "1"}, "SECRET_JWT_KEY", expires=5)
    payload = decode_jwt_token(token, "SECRET_JWT_KEY")
    assert payload["sub"] == "1"


def test_decode_jwt_token_without_exp_check_accepts_expired():
    token = create_jwt_token({"sub": "1"}, "SECRET_JWT_KEY", expires=-1)
    payload = decode_jwt_token_without_exp_check(token, "SECRET_JWT_KEY")
    assert payload["sub"] == "1"


def test_decode_jwt_token_invalid_raises():
    with pytest.raises(HTTPException):
        decode_jwt_token("not-a-token", "SECRET_JWT_KEY")


def test_decode_jwt_token_expired_raises():
    token = create_jwt_token({"sub": "1"}, "SECRET_JWT_KEY", expires=-1)
    with pytest.raises(HTTPException) as exc:
        decode_jwt_token(token, "SECRET_JWT_KEY")
    assert exc.value.status_code == 401


def test_decode_reset_password_expired_raises():
    secret = "reset-secret"
    expired_payload = {"sub": "1", "exp": datetime.now(timezone.utc) - timedelta(minutes=5)}
    token = jwt.encode(expired_payload, secret, algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        decode_jwt_token_reset_password(token, secret)
    assert exc.value.status_code == 401

