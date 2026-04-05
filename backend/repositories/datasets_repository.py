from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Dataset


class DatasetsRepository:
    async def get_by_name(self, db: AsyncSession, name: str) -> Dataset | None:
        result = await db.execute(select(Dataset).filter(Dataset.name == name))
        return result.scalars().first()

    async def get_by_id(self, db: AsyncSession, dataset_id: int) -> Dataset | None:
        result = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
        return result.scalars().first()

    async def get_all(self, db: AsyncSession) -> List[Dataset]:
        result = await db.execute(select(Dataset))
        return list(result.scalars().all())