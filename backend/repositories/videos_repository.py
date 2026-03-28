from typing import List, Sequence

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from models import Dataset, Video


class VideosRepository:
    async def get_videos_paged(
            self,
            db: AsyncSession,
            search_term: str | None,
            dataset_id: Dataset | None = None,
            is_violent: bool | None = None,
            asc: bool = True,
            page: int = 0,
            page_size: int = 40
    ) -> Sequence[Video]:
        query = select(Video)

        if dataset_id is not None:
            query = query.where(Video.dataset_id == dataset_id.value)

        if search_term is not None:
            search_lower = search_term.lower()
            conditions = [Video.name.ilike(f"%{search_term}%")]
            
            matching_dataset_ids = [ds.value for ds in Dataset if search_lower in ds.name.lower()]
            if matching_dataset_ids:
                conditions.append(Video.dataset_id.in_(matching_dataset_ids))
                
            if (
                "non-violent" in search_lower or
                "non violent" in search_lower or
                "nonviolent" in search_lower or
                "nonviolence" in search_lower or
                "non-violence" in search_lower or
                "nonviolence" in search_lower
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
