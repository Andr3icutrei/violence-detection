from typing import List

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import cv2
import tempfile
import os

from sqlalchemy.orm import contains_eager

from models import Dataset, Video
from models.dataset_status import DatasetStatus


class DatasetsRepository:
    async def get_by_name(self, db: AsyncSession, name: str) -> Dataset | None:
        result = await db.execute(select(Dataset).filter(Dataset.name == name))
        return result.scalars().first()

    async def get_by_id(self, db: AsyncSession, dataset_id: int) -> Dataset | None:
        result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
        return result.scalars().first()

    async def get_all_accepted(self, db: AsyncSession) -> List[Dataset]:
        result = await db.execute(select(Dataset).filter(Dataset.status == DatasetStatus.ACCEPTED))
        return list(result.scalars().all())

    async def get_all_pending(self, db: AsyncSession, search_term: str, page: int, page_size: int) -> List[Dataset]:
        query = (select(Dataset)
             .filter(Dataset.status == DatasetStatus.PENDING)
             .join(Dataset.created_by_user)
             .join(Dataset.videos)
             .options(contains_eager(Dataset.created_by_user)))
        if search_term is not None or search_term != "":
            query = query.filter(Dataset.name.lower().contains(search_term.lower()))
        if page is not None and page_size is not None:
            query = query.offset(page * page_size).limit(page_size)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def create_unofficial_dataset(self,
        db: AsyncSession,
        name: str,
        videos: List[UploadFile],
        user_id: int
    ) -> Dataset:

        video_models = []
        for video in videos:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
            try:

                await video.seek(0)
                video_bytes = await video.read()
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(video_bytes)

                cap = cv2.VideoCapture(temp_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = frame_count / fps if fps > 0 else 0
                cap.release()

            finally:
                os.remove(temp_path)

            video_models.append(
                Video(
                    name=video.filename,
                    path=f"{name}/{video.filename}",
                    duration=duration_sec,
                    frame_rate=fps
                )
            )

        dataset = Dataset(
            name=name,
            is_official=False,
            status=DatasetStatus.PENDING,
            videos=video_models,
            created_by_user_id=user_id
        )

        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        return dataset

    async def user_has_pending_datasets(self, db: AsyncSession, user_id: int) -> bool:
        result = await db.execute(select(Dataset).filter(Dataset.created_by_user_id == user_id, Dataset.status == DatasetStatus.PENDING))
        return result.scalars().first() is not None