from typing import Sequence
import uuid

from sqlalchemy import select, or_, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager

from models import Dataset, Video
from models.inference_history import InferenceHistory
from models.inference_history_classification import InferenceHistoryClassification
from models.inference_history_people_tracking import InferenceHistoryPeopleTracking


class VideosRepository:
    async def get_videos_paged(
            self,
            search_term: str | None,
            dataset_id: int | None = None,
            is_violent: bool | None = None,
            asc: bool | None = None,
            page: int = 0,
            page_size: int = 40,
            *,
            db: AsyncSession,
    ) -> Sequence[Video]:
        query = select(Video).join(Video.dataset).options(contains_eager(Video.dataset))

        if dataset_id is not None:
            query = query.where(Video.dataset_id == dataset_id)

        if search_term is not None:
            search_lower = search_term.lower()

            conditions = [
                Video.name.ilike(f"%{search_term}%"),
                Dataset.name.ilike(f"%{search_term}%")
            ]

            if (
                "non-violent" in search_lower or
                "non violent" in search_lower or
                "nonviolent" in search_lower or
                "nonviolence" in search_lower or
                "non-violence" in search_lower
            ):
                conditions.append(Video.is_violent == False)
            elif "violent" in search_lower or "violence" in search_lower:
                conditions.append(Video.is_violent == True)

            query = query.where(or_(*conditions))

        if is_violent is not None:
            query = query.where(Video.is_violent == is_violent)

        query = query.order_by(Video.name.asc() if asc else Video.name.desc())
        query = query.offset(page * page_size).limit(page_size)

        result = await db.execute(query)
        return result.scalars().all()

    async def get_by_uid(self, video_uid: str | int, db: AsyncSession) -> Video | None:
        try:
            parsed_uid = uuid.UUID(str(video_uid))
        except ValueError:
            return None

        result = await db.execute(select(Video).filter(Video.uid == parsed_uid))
        return result.scalars().first()

    async def get_by_id(self, video_id: int, db: AsyncSession) -> Video | None:
        result = await db.execute(select(Video).filter(Video.id == video_id))
        return result.scalars().first()

    async def get_by_id_for_classification(self, video_id: int, db: AsyncSession) -> Video | None:
        query = select(Video).join(Video.dataset).options(contains_eager(Video.dataset)).filter(Video.id == video_id)
        result = await db.execute(query)
        return result.scalars().first()

    async def delete(self, video: Video, db: AsyncSession) -> None:
        history_result = await db.execute(select(InferenceHistory.id).where(InferenceHistory.video_id == video.id))
        history_ids = history_result.scalars().all()

        if history_ids:
            await db.execute(delete(InferenceHistoryClassification).where(InferenceHistoryClassification.inference_history_id.in_(history_ids)))
            await db.execute(delete(InferenceHistoryPeopleTracking).where(InferenceHistoryPeopleTracking.inference_history_id.in_(history_ids)))
            await db.execute(delete(InferenceHistory).where(InferenceHistory.id.in_(history_ids)))

        await db.delete(video)
        await db.flush()

    async def save(self, video: Video, db: AsyncSession) -> Video:
        db.add(video)
        await db.commit()
        await db.refresh(video)
        return video
