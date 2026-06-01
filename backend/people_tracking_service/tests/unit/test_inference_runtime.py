import pytest

from services import inference_runtime


class DummyYOLO:
    def __init__(self, model_path):
        self.model_path = model_path


def test_load_inference_runtime(monkeypatch):
    monkeypatch.setenv("YOLO_MODEL_PATH", "model.pt")
    monkeypatch.setattr(inference_runtime, "YOLO", DummyYOLO)

    runtime = inference_runtime.load_inference_runtime()

    assert runtime.yolo_model_path == "model.pt"
    assert isinstance(runtime.yolo_model, DummyYOLO)
    assert runtime.yolo_model.model_path == "model.pt"

