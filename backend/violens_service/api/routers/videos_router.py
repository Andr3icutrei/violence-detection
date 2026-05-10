from typing import List
import os

from shared_models import InferenceModel
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import get_videos_service
from models import User, Video
from schemas.videos_schema import VideoResponseDto
from services.auth_service import get_current_user
from services.videos_service import VideosService

router = APIRouter(
    prefix="/videos",
    tags=["Videos"],
)

def _cleanup_temp_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)

def _format_score(value: float | str) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid score value returned by classification service.",
        ) from exc

def _media_type_for_path(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".avi":
        return "video/x-msvideo"
    return "video/mp4"

@router.get("/get_videos_paged", response_model=List[VideoResponseDto], status_code=HTTP_200_OK)
async def get_videos_paged(
    current_user: User = Depends(get_current_user),
    asc: bool = True,
    page: int = 0,
    page_size: int = 40,
    is_violent: bool | None = None,
    search_term: str | None = None,
    dataset_id: int | None = None,
    videos_service: VideosService = Depends(get_videos_service),
):
    videos: List[Video] = await videos_service.get_videos_paged(search_term, dataset_id, is_violent, asc, page, page_size)
    return [
        VideoResponseDto(
            id=video.id,
            uid=str(video.uid),
            name=video.name,
            path=video.path,
            is_violent=video.is_violent,
            dataset_id=video.dataset_id,
            dataset_name=video.dataset.name,
            duration=video.duration,
            frame_rate=int(video.frame_rate),
            dataset_is_official=video.dataset.is_official
        ) for video in videos
    ]

@router.get("/exists_video/{video_uid}", response_model=None, status_code=HTTP_200_OK)
async def exists_video(
    video_uid: str,
    current_user: User = Depends(get_current_user),
    videos_service: VideosService = Depends(get_videos_service),
):
    exists: bool = await videos_service.exists_video(video_uid)
    if not exists:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND
        )

@router.post("/classify_video_gradcam/{video_id}", status_code=HTTP_200_OK)
async def inference_video(
    video_id: int,
    background_tasks: BackgroundTasks,
    videos_service: VideosService = Depends(get_videos_service),
    current_user: User = Depends(get_current_user),
):
    inference_result = await videos_service.classify_and_gradcam_video(video_id, current_user)
    background_tasks.add_task(_cleanup_temp_file, inference_result.video_path)

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

@router.post("/people_tracking/{video_id}", status_code=HTTP_200_OK)
async def people_tracking(
    video_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    videos_service: VideosService = Depends(get_videos_service),
):
    try:
        processed_video_path, tracked_count = await videos_service.people_tracking(video_id, current_user)
        background_tasks.add_task(_cleanup_temp_file, processed_video_path)
        return FileResponse(
            path=processed_video_path,
            media_type=_media_type_for_path(processed_video_path),
            filename=f"people_tracking_output{os.path.splitext(processed_video_path)[1]}",
            headers={
                "X-Tracked-People-Count": str(tracked_count),
            },
        )
    except HTTPException:
        raise
