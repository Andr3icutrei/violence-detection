from fastapi import Depends

from core.dependencies.inference_runtime import get_inference_runtime
from services.classification_service import ClassificationService
from services.inference_runtime import InferenceRuntime


def get_classification_service(
    inference_runtime: InferenceRuntime = Depends(get_inference_runtime),
) -> ClassificationService:
    return ClassificationService(inference_runtime)
