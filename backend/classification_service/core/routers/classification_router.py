import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from shared_models import InferenceModel

from starlette.status import HTTP_200_OK

from core.dependencies.classification_service import get_classification_service
from schemas.classification import ClassificationResponseDto
from services.classification_service import ClassificationService

router = APIRouter(
    prefix="/classification",
    tags=["Classification"],
)

@router.get("/classify_video_gradcam", status_code=HTTP_200_OK)
async def inference_video(
    video_path: str,
    inference_model: int,
    videos_service: ClassificationService = Depends(get_classification_service),
) -> ClassificationResponseDto:
    try:
        model_enum = InferenceModel(inference_model)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid inference_model '{inference_model}'. Expected one of: 0, 10, 20.",
        ) from exc
    inference_result = await videos_service.classify_and_gradcam_video(video_path, model_enum)
    return inference_result
