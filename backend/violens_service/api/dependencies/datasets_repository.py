from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from repositories.datasets_repository import DatasetsRepository


def get_datasets_repository(db: AsyncSession = Depends(get_db)) -> DatasetsRepository:
    return DatasetsRepository(db)

