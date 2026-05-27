import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from models.action import Action
from services import videos_service
from services.videos_service import VideosService


def run(coro):
    return asyncio.run(coro)


def make_user(**overrides):
    data = {"id": 1, "credits": 10}
    data.update(overrides)
    return SimpleNamespace(**data)


def make_video(**overrides):
    data = {
        "id": 10,
        "is_violent": True,
        "path": "/tmp/video.mp4",
        "dataset": SimpleNamespace(inference_model_id=1),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def make_service(**repos):
    return VideosService(
        repos.get("videos_repository") or Mock(),
        repos.get("inference_history_repository") or Mock(),
        repos.get("users_repository") or Mock(),
        repos.get("inference_actions_repository") or Mock(),
        repos.get("inference_models_repository") or Mock(),
    )


def test_video_suffix():
    assert VideosService._video_suffix("video/avi") == ".avi"
    assert VideosService._video_suffix("video/mp4") == ".mp4"


def test_parse_classification_headers_missing():
    with pytest.raises(HTTPException) as exc:
        VideosService._parse_classification_headers({})

    assert exc.value.status_code == 500


def test_parse_people_tracking_headers_missing():
    with pytest.raises(HTTPException) as exc:
        VideosService._parse_people_tracking_headers({})

    assert exc.value.status_code == 500


def test_ensure_credits_raises():
    service = make_service()
    user = make_user(credits=1)
    action = SimpleNamespace(credits=5)

    with pytest.raises(HTTPException) as exc:
        run(service._ensure_credits(user, action))

    assert exc.value.status_code == 402


def test_deduct_credits_updates_user():
    repo = Mock()
    repo.add_user = AsyncMock(return_value=None)
    service = make_service(users_repository=repo)
    user = make_user(credits=10)
    action = SimpleNamespace(credits=4)

    run(service._deduct_credits(user, action))

    assert user.credits == 6
    repo.add_user.assert_awaited_once_with(user)


def test_get_inference_model_missing_id():
    service = make_service(inference_models_repository=Mock())

    with pytest.raises(HTTPException) as exc:
        run(service._get_inference_model_or_500(make_video(dataset=SimpleNamespace(inference_model_id=None))))

    assert exc.value.status_code == 500


def test_classify_and_occlusion_success(monkeypatch):
    users_repo = Mock()
    videos_repo = Mock()
    actions_repo = Mock()
    history_repo = Mock()
    models_repo = Mock()

    db_user = make_user(credits=10)
    video = make_video(dataset=SimpleNamespace(inference_model_id=3))
    model_record = SimpleNamespace(path="/models/model.onnx")
    inference_action = SimpleNamespace(credits=2)

    users_repo.get_by_id = AsyncMock(return_value=db_user)
    users_repo.add_user = AsyncMock(return_value=None)
    videos_repo.get_by_id_for_classification = AsyncMock(return_value=video)
    actions_repo.get_inference_action_by_action_id = AsyncMock(return_value=inference_action)
    models_repo.get_by_id = AsyncMock(return_value=model_record)
    history_repo.add_inference_history_classification = AsyncMock(return_value=None)

    service = make_service(
        users_repository=users_repo,
        videos_repository=videos_repo,
        inference_actions_repository=actions_repo,
        inference_history_repository=history_repo,
        inference_models_repository=models_repo,
    )

    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    async def fake_fetch(*_args, **_kwargs):
        headers = {
            "x-predicted-label": "1",
            "x-confidence": "0.9",
            "x-predicted-class-probability": "0.8",
        }
        return temp_path, headers

    monkeypatch.setattr(videos_service, "get_env_variable", lambda *_: "http://service")
    monkeypatch.setattr(service, "_fetch_streamed_video", fake_fetch)

    try:
        result = run(service.classify_and_occlusion_video(10, make_user(id=1)))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    assert result.predicted_label == "1"
    assert result.confidence == 0.9
    assert db_user.credits == 8


def test_people_tracking_success(monkeypatch):
    users_repo = Mock()
    videos_repo = Mock()
    actions_repo = Mock()
    history_repo = Mock()

    db_user = make_user(credits=10)
    video = make_video()
    inference_action = SimpleNamespace(credits=3)

    users_repo.get_by_id = AsyncMock(return_value=db_user)
    users_repo.add_user = AsyncMock(return_value=None)
    videos_repo.get_by_id = AsyncMock(return_value=video)
    actions_repo.get_inference_action_by_action_id = AsyncMock(return_value=inference_action)
    history_repo.add_inference_history_classification = AsyncMock(return_value=None)

    service = make_service(
        users_repository=users_repo,
        videos_repository=videos_repo,
        inference_actions_repository=actions_repo,
        inference_history_repository=history_repo,
    )

    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    async def fake_fetch(*_args, **_kwargs):
        headers = {"x-tracked-people-count": "4"}
        return temp_path, headers

    monkeypatch.setattr(videos_service, "get_env_variable", lambda *_: "http://service")
    monkeypatch.setattr(service, "_fetch_streamed_video", fake_fetch)

    try:
        result_path, tracked = run(service.people_tracking(10, make_user(id=1)))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    assert result_path == temp_path
    assert tracked == 4
    assert db_user.credits == 7

