import os
import asyncio

import pytest
from fastapi import HTTPException

from services.people_tracking import PeopleTrackingService
from services.inference_runtime import InferenceRuntime


def run_async(coro):
    return asyncio.run(coro)


def test_people_tracking_download_not_found(monkeypatch):
    runtime = InferenceRuntime(yolo_model_path="dummy.pt", yolo_model=object())
    service = PeopleTrackingService(inference_runtime=runtime)

    async def _download(*args, **kwargs):
        return False

    monkeypatch.setattr("services.people_tracking.download_object_to_file", _download)

    with pytest.raises(HTTPException) as excinfo:
        run_async(service.people_tracking("missing.mp4"))

    assert excinfo.value.status_code == 500
    assert "Video file not found in storage" in str(excinfo.value.detail)


def test_people_tracking_success(monkeypatch, tmp_path):
    runtime = InferenceRuntime(yolo_model_path="dummy.pt", yolo_model=object())
    service = PeopleTrackingService(inference_runtime=runtime)
    overlay_path = str(tmp_path / "overlay.mp4")

    async def _download(_key, target_path):
        with open(target_path, "wb") as handle:
            handle.write(b"data")
        return True

    def _run_people_tracking(_temp_path, _model):
        with open(overlay_path, "wb") as handle:
            handle.write(b"overlay")
        return overlay_path, 3

    monkeypatch.setattr("services.people_tracking.download_object_to_file", _download)
    monkeypatch.setattr("services.people_tracking.run_people_tracking", _run_people_tracking)

    result_path, people_tracked, temp_path = run_async(service.people_tracking("video.mp4"))

    assert result_path == overlay_path
    assert people_tracked == 3
    assert os.path.exists(temp_path)

    os.remove(temp_path)


def test_people_tracking_error_cleanup(monkeypatch):
    runtime = InferenceRuntime(yolo_model_path="dummy.pt", yolo_model=object())
    service = PeopleTrackingService(inference_runtime=runtime)

    async def _download(*args, **kwargs):
        return True

    def _run_people_tracking(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("services.people_tracking.download_object_to_file", _download)
    monkeypatch.setattr("services.people_tracking.run_people_tracking", _run_people_tracking)

    with pytest.raises(HTTPException) as excinfo:
        run_async(service.people_tracking("video.mp4"))

    assert excinfo.value.status_code == 500
