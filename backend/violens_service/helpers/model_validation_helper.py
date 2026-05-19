from typing import Iterable

import onnx
from fastapi import HTTPException
from starlette import status

EXPECTED_INPUT_DIMS = (3, 16, 112, 112)
EXPECTED_OUTPUT_CLASSES = 2


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
            detail=f"Invalid ONNX model: {exc}",
        ) from exc

    if not model.graph.input:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ONNX model has no inputs.",
        )
    if not model.graph.output:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ONNX model has no outputs.",
        )

    input_tensor = model.graph.input[0]
    input_shape = input_tensor.type.tensor_type.shape
    input_dims = _read_dims(input_shape.dim)
    if len(input_dims) != 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ONNX input must be 5D (N, C, T, H, W) for a 3D CNN.",
        )
    expected = EXPECTED_INPUT_DIMS
    for actual_dim, expected_dim in zip(input_dims[1:], expected):
        if actual_dim is not None and actual_dim != expected_dim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ONNX input shape must match (N, 3, 16, 112, 112).",
            )

    output_tensor = model.graph.output[0]
    output_shape = output_tensor.type.tensor_type.shape
    output_dims = _read_dims(output_shape.dim)
    if len(output_dims) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ONNX output must be 2D (N, num_classes).",
        )
    if output_dims[1] is not None and output_dims[1] != EXPECTED_OUTPUT_CLASSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ONNX output must have 2 classes like the reference 3D CNN.",
        )
