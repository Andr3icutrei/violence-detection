from typing import Sequence
import uuid

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager

from models import Dataset, Video


class VideosRepository:
    async def get_videos_paged(
            self,
            db: AsyncSession,
            search_term: str | None,
            dataset_id: int | None = None,
            is_violent: bool | None = None,
            asc: bool | None = None,
            page: int = 0,
            page_size: int = 40
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

    async def get_by_uid(self, db: AsyncSession, video_uid: str | int) -> Video | None:
        try:
            parsed_uid = uuid.UUID(str(video_uid))
        except ValueError:
            return None

        result = await db.execute(select(Video).filter(Video.uid == parsed_uid))
        return result.scalars().first()

    async def get_by_id(self, db: AsyncSession, video_id: int) -> Video | None:
        result = await db.execute(select(Video).filter(Video.id == video_id))
        return result.scalars().first()

    async def get_by_id_for_classification(self, db: AsyncSession, video_id: int) -> Video | None:
        query = select(Video).join(Video.dataset).options(contains_eager(Video.dataset)).filter(Video.id == video_id)
        result = await db.execute(query)
        return result.scalars().first()
