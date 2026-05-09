from dataclasses import dataclass
from typing import Any
from ultralytics import YOLO

from helpers.env_helper import get_env_variable


def _resolve_symbol(module: Any, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"None of the symbols {names} exist in module '{module.__name__}'.")


@dataclass
class InferenceRuntime:
    yolo_model_path: str
    yolo_model: Any


def load_inference_runtime() -> InferenceRuntime:
    yolo_model_path = get_env_variable("YOLO_MODEL_PATH")
    yolo_model = YOLO(yolo_model_path)

    return InferenceRuntime(
        yolo_model_path=yolo_model_path,
        yolo_model=yolo_model,
    )
