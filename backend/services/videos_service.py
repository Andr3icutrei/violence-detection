import asyncio
from typing import List, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from helpers.bucket_helper import get_presigned_url
from models import Dataset, Video
from repositories.users_repository import UsersRepository
from repositories.videos_repository import VideosRepository


class VideosService:
    def __init__(self):
        self.videos_repository = VideosRepository()

    async def get_videos_paged(
            self,
            db: AsyncSession,
            search_term: str | None,
            dataset_id: int | None = None,
            is_violent: bool | None = None,
            asc: bool = True,
            page: int = 0,
            page_size: int = 40
    ) -> List[Video]:

        videos: Sequence[Video] = await (
            self.videos_repository.get_videos_paged(db, search_term, dataset_id, is_violent, asc, page, page_size))

        tasks = [get_presigned_url(video.path) for video in videos]
        presigned_urls = await asyncio.gather(*tasks)

        result = []
        for video, url in zip(videos, presigned_urls):
            video.path = url
            result.append(video)

        return result

    async def exists_video(self, db: AsyncSession, video_uid: str) -> bool:
        video: Video = await self.videos_repository.get_by_uid(db, video_uid)
        return video is not None