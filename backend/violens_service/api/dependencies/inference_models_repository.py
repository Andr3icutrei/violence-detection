from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from repositories.inference_models_repository import InferenceModelsRepository


def get_inference_models_repository(
    db: AsyncSession = Depends(get_db),
) -> InferenceModelsRepository:
    return InferenceModelsRepository(db)

