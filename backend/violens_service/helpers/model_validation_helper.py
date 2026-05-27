from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from fastapi import HTTPException
from starlette import status

EXPECTED_OUTPUT_CLASSES = 2
DEFAULT_DUMMY_INPUT_DIMS = (3, 16, 112, 112)


def _dim_value(dim) -> int | None:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    return None


def _read_dims(dims: Iterable) -> list[int | None]:
    return [_dim_value(dim) for dim in dims]


def validate_3dcnn_onnx(model_path: str) -> None:
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_INVALID",
                "message": f"Invalid ONNX model: {exc}",
            },
        ) from exc

    if not model.graph.input:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_NO_INPUTS",
                "message": "ONNX model has no inputs.",
            },
        )
    if not model.graph.output:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_NO_OUTPUTS",
                "message": "ONNX model has no outputs.",
            },
        )

    input_tensor = model.graph.input[0]
    input_shape = input_tensor.type.tensor_type.shape
    input_dims = _read_dims(input_shape.dim)
    if len(input_dims) != 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_INPUT_NOT_5D",
                "message": "ONNX input must be 5D (N, C, T, H, W) for a 3D CNN.",
            },
        )

    output_tensor = model.graph.output[0]
    output_shape = output_tensor.type.tensor_type.shape
    output_dims = _read_dims(output_shape.dim)
    if len(output_dims) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_OUTPUT_NOT_2D",
                "message": "ONNX output must be 2D (N, num_classes).",
            },
        )
    if output_dims[1] is not None and output_dims[1] != EXPECTED_OUTPUT_CLASSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_OUTPUT_NOT_BINARY",
                "message": "ONNX output must have 2 classes like the reference 3D CNN.",
            },
        )

    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_INFERENCE_LOAD_FAILED",
                "message": f"Failed to load ONNX model for inference: {exc}",
            },
        ) from exc

    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_INPUT_COUNT_INVALID",
                "message": "ONNX model must have exactly one input tensor.",
            },
        )

    dummy_dims = [1]
    for dim_value, fallback in zip(input_dims[1:], DEFAULT_DUMMY_INPUT_DIMS):
        dummy_dims.append(dim_value if dim_value is not None else fallback)

    dummy_input = np.zeros(tuple(dummy_dims), dtype=np.float32)
    try:
        outputs = session.run(None, {inputs[0].name: dummy_input})
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_DUMMY_INFERENCE_FAILED",
                "message": f"ONNX model failed dummy inference: {exc}",
            },
        ) from exc

    if not outputs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_NO_OUTPUTS_RUNTIME",
                "message": "ONNX model returned no outputs.",
            },
        )

    output = outputs[0]
    if getattr(output, "ndim", None) != 2 or output.shape[1] != EXPECTED_OUTPUT_CLASSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "ONNX_OUTPUT_SHAPE_INVALID",
                "message": "ONNX output must be (N, 2) for binary classification.",
            },
        )
