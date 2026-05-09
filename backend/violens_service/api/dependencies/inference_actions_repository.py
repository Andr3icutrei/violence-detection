from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from repositories.inference_actions_repository import InferenceActionsRepository


def get_inference_actions_repository(
    db: AsyncSession = Depends(get_db),
) -> InferenceActionsRepository:
    return InferenceActionsRepository(db)

