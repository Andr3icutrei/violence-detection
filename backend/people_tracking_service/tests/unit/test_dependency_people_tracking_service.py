from core.dependencies.people_tracking_service import get_classification_service
from services.inference_runtime import InferenceRuntime
from services.people_tracking import PeopleTrackingService


def test_get_classification_service_returns_service():
    runtime = InferenceRuntime(yolo_model_path="dummy.pt", yolo_model=object())

    service = get_classification_service(inference_runtime=runtime)

    assert isinstance(service, PeopleTrackingService)
    assert service.inference_runtime is runtime

