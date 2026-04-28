from typing import List, Tuple

from fastapi import UploadFile
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import cv2
import tempfile
import os

from sqlalchemy.orm import contains_eager

from models import Dataset, Video, InferenceHistory
from models.dataset_status import DatasetStatus
from models.inference_history_classification import InferenceHistoryClassification
from models.inference_history_people_tracking import InferenceHistoryPeopleTracking


class DatasetsRepository:
    async def get_by_name(self, name: str, db: AsyncSession) -> Dataset | None:
        result = await db.execute(select(Dataset).filter(Dataset.name == name))
        return result.scalars().first()

    async def get_by_id(self, dataset_id: int, db: AsyncSession) -> Dataset | None:
        result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
        return result.scalars().first()

    async def get_all_accepted(self, db: AsyncSession) -> List[Dataset]:
        result = await db.execute(select(Dataset).filter(Dataset.status == DatasetStatus.ACCEPTED))
        return list(result.scalars().all())

    async def get_all(self, search_term: str | None, page: int, page_size: int, dataset_status: DatasetStatus | None, db: AsyncSession) -> List[Dataset]:
        query = (select(Dataset)
             .filter(Dataset.is_official == False)
             .join(Dataset.created_by_user)
             .join(Dataset.videos)
             .options(
                 contains_eager(Dataset.created_by_user),
                 contains_eager(Dataset.videos)
             ))
        if dataset_status is not None:
            query = query.filter(Dataset.status == dataset_status)
        if search_term:
            query = query.filter(Dataset.name.ilike(f"%{search_term}%"))
        if page is not None and page_size is not None:
            query = query.offset(page * page_size).limit(page_size)
        result = await db.execute(query)
        return list(result.scalars().unique().all())

    async def create_unofficial_dataset(self,
        name: str,
        videos: List[UploadFile],
        user_id: int,
        db: AsyncSession,
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

    async def user_has_pending_datasets(self, user_id: int, db: AsyncSession) -> bool:
        result = await db.execute(select(Dataset).filter(Dataset.created_by_user_id == user_id, Dataset.status == DatasetStatus.PENDING))
        return result.scalars().first() is not None

    async def get_by_id_with_videos(self, dataset_id: int, db: AsyncSession) -> Dataset | None:
        result = await db.execute(
            select(Dataset)
            .filter(Dataset.id == dataset_id)
            .join(Dataset.videos)
            .join(Dataset.created_by_user)
            .options(
                contains_eager(Dataset.created_by_user),
                contains_eager(Dataset.videos)
            )
        )
        return result.scalars().first()

    async def save(self, dataset: Dataset, db: AsyncSession) -> Dataset:
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        return dataset

    async def delete(self, dataset: Dataset, db: AsyncSession) -> None:
        await db.delete(dataset)
        await db.flush()

    async def get_most_popular_dataset_classification(self, db: AsyncSession) -> Tuple[Dataset, int] | None:
        inferences_count = func.coalesce(func.count(InferenceHistoryClassification.id), 0)
        result = await db.execute(
            select(Dataset, inferences_count)
            .join(Dataset.videos)
            .join(Video.inference_history)
            .join(InferenceHistory.inference_history_classification)
            .group_by(Dataset.id)
            .order_by(inferences_count.desc())
        )
        return result.first()

    async def get_most_popular_dataset_people_tracking(self, db: AsyncSession) -> Tuple[Dataset, int] | None:
        inferences_count = func.coalesce(func.count(InferenceHistoryPeopleTracking.id), 0)
        result = await db.execute(
            select(Dataset, inferences_count)
            .join(Dataset.videos)
            .join(Video.inference_history)
            .join(InferenceHistory.inference_history_people_tracking)
            .group_by(Dataset.id)
            .order_by(inferences_count.desc())
        )
        return result.first()

    async def get_official_datasets_count(self, db: AsyncSession) -> int:
        result = await db.execute(select(func.count(Dataset.id)).filter(Dataset.is_official == True))
        return result.scalar_one()

    async def get_unofficial_datasets_count(self, db: AsyncSession) -> int:
        result = await db.execute(select(func.count(Dataset.id)).filter(Dataset.is_official == False))
        return result.scalar_one()

    async def get_pending_datasets_count(self, db: AsyncSession) -> int:
        result = await db.execute(select(func.count(Dataset.id)).filter(Dataset.status == DatasetStatus.PENDING))
        return result.scalar_one()