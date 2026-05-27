import os

import pytest
from fastapi.testclient import TestClient

from tests.fakes import (
    FakeAuthService,
    FakeCreditsService,
    FakeDatasetsService,
    FakeInferenceActionsService,
    FakeInferenceHistoryService,
    FakeUsersService,
    FakeVideosService,
    make_user,
)


os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/db")
os.environ.setdefault("MAIL_USERNAME", "test@example.com")
os.environ.setdefault("MAIL_PASSWORD", "password")


@pytest.fixture()
def app():
    from main import app as fastapi_app
    from api.dependencies import (
        get_auth_service,
        get_credits_service,
        get_datasets_service,
        get_inference_actions_service,
        get_inference_history_service,
        get_users_service,
        get_videos_service,
    )
    from services.auth_service import get_current_admin_user, get_current_user

    fastapi_app.dependency_overrides[get_auth_service] = lambda: FakeAuthService()
    fastapi_app.dependency_overrides[get_credits_service] = lambda: FakeCreditsService()
    fastapi_app.dependency_overrides[get_datasets_service] = lambda: FakeDatasetsService()
    fastapi_app.dependency_overrides[get_inference_actions_service] = lambda: FakeInferenceActionsService()
    fastapi_app.dependency_overrides[get_inference_history_service] = lambda: FakeInferenceHistoryService()
    fastapi_app.dependency_overrides[get_users_service] = lambda: FakeUsersService()
    fastapi_app.dependency_overrides[get_videos_service] = lambda: FakeVideosService()
    fastapi_app.dependency_overrides[get_current_user] = lambda: make_user()
    fastapi_app.dependency_overrides[get_current_admin_user] = lambda: make_user(is_admin=True)

    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest.fixture()
def client(app):
    with TestClient(app) as client_instance:
        yield client_instance

