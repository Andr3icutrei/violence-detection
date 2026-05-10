import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse

from starlette.status import HTTP_200_OK

from core.dependencies.people_tracking_service import get_classification_service
from schemas.people_tracking import PeopleTrackingResponseDto
from services.people_tracking import PeopleTrackingService

router = APIRouter(
    prefix="/people_tracking",
    tags=["People Tracking"],
)

def _cleanup_temp_file(file_path: str) -> None:
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

def _media_type_for_path(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".avi":
        return "video/x-msvideo"
    return "video/mp4"

@router.get("", response_model=PeopleTrackingResponseDto, status_code=HTTP_200_OK)
async def inference_video(
    video_path: str,
    people_tracking_service: PeopleTrackingService = Depends(get_classification_service),
) -> PeopleTrackingResponseDto:
    overlay_video_path, people_tracked, _ = await people_tracking_service.people_tracking(video_path)
    return PeopleTrackingResponseDto(
        video_path=overlay_video_path,
        people_tracked=str(people_tracked)
    )


@router.get("/stream", status_code=HTTP_200_OK)
async def inference_video_stream(
    video_path: str,
    background_tasks: BackgroundTasks,
    people_tracking_service: PeopleTrackingService = Depends(get_classification_service),
) -> FileResponse:
    overlay_video_path, people_tracked, temp_video_path = await people_tracking_service.people_tracking(video_path)
    background_tasks.add_task(_cleanup_temp_file, overlay_video_path)
    background_tasks.add_task(_cleanup_temp_file, temp_video_path)

    return FileResponse(
        path=overlay_video_path,
        media_type=_media_type_for_path(overlay_video_path),
        filename=f"people_tracking_output{os.path.splitext(overlay_video_path)[1]}",
        headers={
            "X-Tracked-People-Count": str(people_tracked),
        },
    )
