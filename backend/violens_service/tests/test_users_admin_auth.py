from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from api.dependencies import get_users_service
from main import app as fastapi_app
from services.auth_service import get_current_admin_user, get_current_user
from tests.fakes import FakeUsersService


def _raise_unauthorized():
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def _raise_forbidden():
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


def test_admin_endpoints_require_admin():
    fastapi_app.dependency_overrides[get_users_service] = lambda: FakeUsersService()
    fastapi_app.dependency_overrides[get_current_admin_user] = _raise_forbidden

    with TestClient(fastapi_app) as client:
        response = client.get("/users/get_all_users")
        assert response.status_code == 403

        response = client.patch("/users/update_user_role", params={"user_id": 1, "is_admin": True})
        assert response.status_code == 403

        response = client.patch("/users/ban_user/1", json={"ban_reason": "spam"})
        assert response.status_code == 403

        response = client.get("/users/get_users_stats")
        assert response.status_code == 403

    fastapi_app.dependency_overrides.clear()


def test_topbar_requires_authenticated_user():
    fastapi_app.dependency_overrides[get_users_service] = lambda: FakeUsersService()
    fastapi_app.dependency_overrides[get_current_user] = _raise_unauthorized

    with TestClient(fastapi_app) as client:
        response = client.get("/users/topbar_information")
        assert response.status_code == 401

    fastapi_app.dependency_overrides.clear()

