import inference.slowfast.config as slowfast_config_module
import inference.slowfast.model as slowfast_model_module
import inference.slowfast.pipeline as slowfast_pipeline_module
import inference.resnet3d.config as resnet3d_config_module
import inference.resnet3d.model as resnet3d_model_module
import inference.resnet3d.pipeline as resnet3d_pipeline_module
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
    slowfast_config: Any
    resnet3d_config: Any
    slowfast_pipeline: Any
    resnet3d_pipeline: Any


def load_inference_runtime() -> InferenceRuntime:
    yolo_model_path = get_env_variable("YOLO_MODEL_PATH")
    yolo_model = YOLO(yolo_model_path)
    slowfast_model_path = get_env_variable("SLOWFAST_MODEL_PATH")
    resnet3d_model_path = get_env_variable("RESNET3D_MODEL_PATH")

    slowfast_config = getattr(slowfast_config_module, "InferenceConfig")()
    resnet3d_config = getattr(resnet3d_config_module, "InferenceConfig")()

    slowfast_pipeline_class = _resolve_symbol(slowfast_pipeline_module, "InferencePipeline")
    slowfast_model_class = _resolve_symbol(slowfast_model_module, "SlowFastViolence")
    resnet3d_pipeline_class = _resolve_symbol(
        resnet3d_pipeline_module,
        "InferencePipelineR3D",
        "InferencePipeline",
    )
    resnet3d_model_class = _resolve_symbol(
        resnet3d_model_module,
        "R3D18Violence",
        "ResNet3DViolence",
    )

    slowfast_pipeline = slowfast_pipeline_class(
        model_path=slowfast_model_path,
        config=slowfast_config,
        model_class=slowfast_model_class,
    )
    resnet3d_pipeline = resnet3d_pipeline_class(
        model_path=resnet3d_model_path,
        config=resnet3d_config,
        model_class=resnet3d_model_class,
    )

    return InferenceRuntime(
        yolo_model_path=yolo_model_path,
        yolo_model=yolo_model,
        slowfast_config=slowfast_config,
        resnet3d_config=resnet3d_config,
        slowfast_pipeline=slowfast_pipeline,
        resnet3d_pipeline=resnet3d_pipeline,
    )
