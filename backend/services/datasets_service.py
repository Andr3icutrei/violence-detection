from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from models import Dataset
from schemas.datasets_schema import DatasetResponseDto


class DatasetsService:
    def __init__(self):
        pass

    async def get_datasets(self) -> List[DatasetResponseDto]:
        return [DatasetResponseDto(id=data.value, name=data.name) for data in Dataset]