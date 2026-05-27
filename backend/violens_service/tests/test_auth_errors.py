from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from api.dependencies import get_auth_service
from main import app as fastapi_app
from services.auth_service import get_current_user


def test_login_invalid_credentials_returns_401():
    class FailingAuthService:
        async def login(self, _email: str, _password: str):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect password.")

    fastapi_app.dependency_overrides[get_auth_service] = lambda: FailingAuthService()

    try:
        with TestClient(fastapi_app) as client:
            response = client.post(
                "/auth/login",
                json={"email": "user@example.com", "password": "bad"},
            )
            assert response.status_code == 401
            assert response.json()["detail"] == "Incorrect password."
    finally:
        fastapi_app.dependency_overrides.clear()


def test_me_expired_token_returns_401():
    def _raise_expired():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")

    fastapi_app.dependency_overrides[get_current_user] = _raise_expired

    try:
        with TestClient(fastapi_app) as client:
            response = client.get("/auth/me")
            assert response.status_code == 401
            assert response.json()["detail"] == "Token has expired"
    finally:
        fastapi_app.dependency_overrides.clear()


def test_logout_expired_token_returns_401():
    def _raise_expired():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")

    fastapi_app.dependency_overrides[get_current_user] = _raise_expired

    try:
        with TestClient(fastapi_app) as client:
            response = client.post("/auth/logout")
            assert response.status_code == 401
            assert response.json()["detail"] == "Token has expired"
    finally:
        fastapi_app.dependency_overrides.clear()

