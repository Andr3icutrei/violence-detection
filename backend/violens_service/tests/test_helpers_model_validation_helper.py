import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

import helpers.model_validation_helper as model_validation_helper


def run(coro):
    return asyncio.run(coro)


class DummyDim:
    def __init__(self, value: int | None):
        self.dim_value = value

    def HasField(self, name: str) -> bool:
        return name == "dim_value" and self.dim_value is not None


class DummyValueInfo:
    def __init__(self, dims: list[int | None]):
        shape = SimpleNamespace(dim=[DummyDim(value) for value in dims])
        tensor_type = SimpleNamespace(shape=shape)
        self.type = SimpleNamespace(tensor_type=tensor_type)


class DummySession:
    def __init__(self, outputs, inputs_count: int = 1) -> None:
        self._outputs = outputs
        self._inputs = [SimpleNamespace(name="input")] * inputs_count

    def get_inputs(self):
        return self._inputs

    def run(self, _unused, _feeds):
        return self._outputs


def build_model(input_dims: list[int | None], output_dims: list[int | None]):
    graph = SimpleNamespace(
        input=[DummyValueInfo(input_dims)],
        output=[DummyValueInfo(output_dims)],
    )
    return SimpleNamespace(graph=graph)


def test_validate_3dcnn_onnx_invalid_load(monkeypatch):
    def _raise(_path):
        raise RuntimeError("bad")

    monkeypatch.setattr(model_validation_helper.onnx, "load", _raise)

    with pytest.raises(HTTPException) as exc:
        model_validation_helper.validate_3dcnn_onnx("model.onnx")

    assert exc.value.detail["error_code"] == "ONNX_INVALID"


def test_validate_3dcnn_onnx_input_not_5d(monkeypatch):
    model = build_model([1, 3, 112, 112], [1, 2])
    monkeypatch.setattr(model_validation_helper.onnx, "load", lambda _path: model)
    monkeypatch.setattr(model_validation_helper.onnx.checker, "check_model", lambda _model: None)

    with pytest.raises(HTTPException) as exc:
        model_validation_helper.validate_3dcnn_onnx("model.onnx")

    assert exc.value.detail["error_code"] == "ONNX_INPUT_NOT_5D"


def test_validate_3dcnn_onnx_output_not_binary(monkeypatch):
    model = build_model([1, 3, 16, 112, 112], [1, 3])
    monkeypatch.setattr(model_validation_helper.onnx, "load", lambda _path: model)
    monkeypatch.setattr(model_validation_helper.onnx.checker, "check_model", lambda _model: None)

    with pytest.raises(HTTPException) as exc:
        model_validation_helper.validate_3dcnn_onnx("model.onnx")

    assert exc.value.detail["error_code"] == "ONNX_OUTPUT_NOT_BINARY"


def test_validate_3dcnn_onnx_runtime_output_shape(monkeypatch):
    model = build_model([1, 3, 16, 112, 112], [1, 2])
    monkeypatch.setattr(model_validation_helper.onnx, "load", lambda _path: model)
    monkeypatch.setattr(model_validation_helper.onnx.checker, "check_model", lambda _model: None)
    monkeypatch.setattr(
        model_validation_helper.ort,
        "InferenceSession",
        lambda *_args, **_kwargs: DummySession(outputs=[np.zeros((1, 3), dtype=np.float32)]),
    )

    with pytest.raises(HTTPException) as exc:
        model_validation_helper.validate_3dcnn_onnx("model.onnx")

    assert exc.value.detail["error_code"] == "ONNX_OUTPUT_SHAPE_INVALID"

