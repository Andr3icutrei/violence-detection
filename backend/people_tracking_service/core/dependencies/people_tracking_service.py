from fastapi.params import Depends

from core.dependencies.inference_runtime import get_inference_runtime
from services.inference_runtime import InferenceRuntime
from services.people_tracking import PeopleTrackingService


def get_classification_service(inference_runtime: InferenceRuntime = Depends(get_inference_runtime)) -> PeopleTrackingService:
    return PeopleTrackingService(inference_runtime=inference_runtime)