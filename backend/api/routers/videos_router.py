from typing import List
import os

from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from core.database import get_db
from models import User, Video
from schemas.videos_schema import VideoResponseDto
from services.auth_service import AuthService
from services.videos_service import VideosService

router = APIRouter(
    prefix="/videos",
    tags=["Videos"],
)

auth_service = AuthService()

def get_videos_service(request: Request) -> VideosService:
    return request.app.state.videos_service

def _cleanup_temp_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)

@router.get("/get_videos_paged", response_model=List[VideoResponseDto], status_code=HTTP_200_OK)
async def get_videos_paged(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(auth_service.get_current_user),
    asc: bool = True,
    page: int = 0,
    page_size: int = 40,
    is_violent: bool | None = None,
    search_term: str | None = None,
    dataset_id: int | None = None,
    videos_service: VideosService = Depends(get_videos_service),
):
    videos: List[Video] = await videos_service.get_videos_paged(search_term, dataset_id, is_violent, asc, page, page_size, db=db)
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
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
    videos_service: VideosService = Depends(get_videos_service),
):
    exists: bool = await videos_service.exists_video(video_uid, db)
    if not exists:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND
        )

@router.post("/inference_video/{video_id}", status_code=HTTP_200_OK)
async def inference_video(
    video_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
    videos_service: VideosService = Depends(get_videos_service),
):
    inference_result = await videos_service.classify_and_gradcam_video(video_id, current_user, db)
    background_tasks.add_task(_cleanup_temp_file, inference_result.video_path)

    formatted_conf = f"{int(inference_result.confidence * 100) / 100:.2f}"
    formatted_prob = f"{int(inference_result.predicted_class_probability * 100) / 100:.2f}"

    return FileResponse(
        path=inference_result.video_path,
        media_type="video/mp4",
        filename="gradcam_output.mp4",
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
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
    videos_service: VideosService = Depends(get_videos_service),
):
    processed_video_path, tracked_count = await videos_service.people_tracking(video_id, current_user, db)
    background_tasks.add_task(_cleanup_temp_file, processed_video_path)
    return FileResponse(
        path=processed_video_path,
        media_type="video/mp4",
        filename="people_tracking_output.mp4",
        headers={
            "X-Tracked-People-Count": str(tracked_count),
        },
    )
