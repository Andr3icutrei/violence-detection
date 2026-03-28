from typing import List

from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import HTTP_200_OK
from fastapi import APIRouter, Depends

from core.database import get_db
from models import Dataset, User, Video
from schemas.videos_schema import VideoResponseDto
from services.auth_service import AuthService
from services.users_service import UsersService
from services.videos_service import VideosService

router = APIRouter(
    prefix="/videos",
    tags=["Videos"],
)

videos_service = VideosService()
auth_service = AuthService()

@router.get("/get_videos_paged", response_model=List[VideoResponseDto], status_code=HTTP_200_OK)
async def get_videos_paged(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(auth_service.get_current_user),
    asc: bool = True,
    page: int = 0,
    page_size: int = 40,
    is_violent: bool | None = None,
    search_term: str | None = None,
    dataset_id: Dataset | None = None,
):
    videos: List[Video] = await videos_service.get_videos_paged(db, search_term, dataset_id, is_violent, asc, page, page_size)
    return [
        VideoResponseDto(
            id=video.id,
            name=video.name,
            path=video.path,
            is_violent=video.is_violent,
            dataset_id=video.dataset_id,
            dataset_name=Dataset(video.dataset_id).name,
            duration=video.duration
        ) for video in videos
    ]

