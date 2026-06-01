import pytest
from fastapi.testclient import TestClient

from services.inference_runtime import InferenceRuntime
import main


class DummyYoloModel:
    def track(self, *args, **kwargs):  # pragma: no cover - used only in integration stubs
        return []


@pytest.fixture
def app(monkeypatch):
    def _load_inference_runtime():
        return InferenceRuntime(yolo_model_path="dummy.pt", yolo_model=DummyYoloModel())

    monkeypatch.setattr(main, "load_inference_runtime", _load_inference_runtime)
    main.app.dependency_overrides = {}
    yield main.app
    main.app.dependency_overrides = {}


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client

