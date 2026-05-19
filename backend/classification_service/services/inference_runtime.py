from dataclasses import dataclass
from typing import Any
from ultralytics import YOLO

from helpers.env_helper import get_env_variable


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
