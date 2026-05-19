import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse

from starlette.status import HTTP_200_OK

from core.dependencies.classification_service import get_classification_service
from schemas.classification import ClassificationResponseDto
from services.classification_service import ClassificationService

router = APIRouter(
    prefix="/classification",
    tags=["Classification"],
)

def _cleanup_temp_file(file_path: str) -> None:
    if file_path and os.path.exists(file_path):
        os.remove(file_path)


def _format_score(value: float | str) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Invalid score value returned by classification service.",
        ) from exc

def _media_type_for_path(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".avi":
        return "video/x-msvideo"
    return "video/mp4"

@router.get("/classify_video_gradcam", status_code=HTTP_200_OK)
async def inference_video(
    video_path: str,
    inference_model_path: str,
    inference_model_kind: str | None = None,
    videos_service: ClassificationService = Depends(get_classification_service),
) -> ClassificationResponseDto:
    inference_result, _ = await videos_service.classify_and_gradcam_video(
        video_path,
        inference_model_path,
        inference_model_kind,
    )
    return inference_result


@router.get("/classify_video_gradcam_stream", status_code=HTTP_200_OK)
async def inference_video_stream(
    video_path: str,
    inference_model_path: str,
    background_tasks: BackgroundTasks,
    inference_model_kind: str | None = None,
    videos_service: ClassificationService = Depends(get_classification_service),
) -> FileResponse:
    inference_result, temp_video_path = await videos_service.classify_and_gradcam_video(
        video_path,
        inference_model_path,
        inference_model_kind,
    )
    background_tasks.add_task(_cleanup_temp_file, inference_result.video_path)
    background_tasks.add_task(_cleanup_temp_file, temp_video_path)

    formatted_conf = _format_score(inference_result.confidence)
    formatted_prob = _format_score(inference_result.predicted_class_probability)

    return FileResponse(
        path=inference_result.video_path,
        media_type=_media_type_for_path(inference_result.video_path),
        filename=f"gradcam_output{os.path.splitext(inference_result.video_path)[1]}",
        headers={
            "X-Predicted-Label": str(inference_result.predicted_label),
            "X-Confidence": formatted_conf,
            "X-Predicted-Class-Probability": formatted_prob,
        },
    )
