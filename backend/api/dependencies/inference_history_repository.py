from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from repositories.inference_history_repository import InferenceHistoryRepository


def get_inference_history_repository(
    db: AsyncSession = Depends(get_db),
) -> InferenceHistoryRepository:
    return InferenceHistoryRepository(db)

