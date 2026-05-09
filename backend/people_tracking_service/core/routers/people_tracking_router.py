import os

from fastapi import APIRouter, Depends


from starlette.status import HTTP_200_OK

from core.dependencies.people_tracking_service import get_classification_service
from schemas.people_tracking import PeopleTrackingResponseDto
from services.people_tracking import PeopleTrackingService

router = APIRouter(
    prefix="/people_tracking",
    tags=["People Tracking"],
)

@router.get("", response_model=PeopleTrackingResponseDto, status_code=HTTP_200_OK)
async def inference_video(
    video_path: str,
    people_tracking_service: PeopleTrackingService = Depends(get_classification_service),
) -> PeopleTrackingResponseDto:
    overlay_video_path, people_tracked = await people_tracking_service.people_tracking(video_path)
    return PeopleTrackingResponseDto(
        video_path=overlay_video_path,
        people_tracked=str(people_tracked)
    )
