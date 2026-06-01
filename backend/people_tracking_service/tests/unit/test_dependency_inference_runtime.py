import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from core.dependencies.inference_runtime import get_inference_runtime
from services.inference_runtime import InferenceRuntime


def _make_request(app):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "app": app,
    }
    return Request(scope)


def test_get_inference_runtime_returns_runtime():
    app = FastAPI()
    runtime = InferenceRuntime(yolo_model_path="dummy.pt", yolo_model=object())
    app.state.inference_runtime = runtime

    request = _make_request(app)

    assert get_inference_runtime(request) is runtime


def test_get_inference_runtime_raises_when_missing():
    app = FastAPI()
    app.state.inference_runtime = None

    request = _make_request(app)

    with pytest.raises(HTTPException) as excinfo:
        get_inference_runtime(request)

    assert excinfo.value.status_code == 503

