from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from repositories.videos_repository import VideosRepository


def get_videos_repository(db: AsyncSession = Depends(get_db)) -> VideosRepository:
    return VideosRepository(db)

