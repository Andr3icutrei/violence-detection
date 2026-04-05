from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Dataset
from repositories.datasets_repository import DatasetsRepository
from schemas.datasets_schema import DatasetResponseDto


class DatasetsService:
    def __init__(self):
        self.datasets_repository = DatasetsRepository()

    async def get_datasets(self, db: AsyncSession) -> List[DatasetResponseDto]:
        result: List[Dataset] = await self.datasets_repository.get_all(db)
        return [
            DatasetResponseDto(
                id=dataset.id,
                name=dataset.name,
                is_official=dataset.is_official
            ) for dataset in result
        ]