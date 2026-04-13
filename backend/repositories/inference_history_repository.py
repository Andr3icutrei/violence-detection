from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import InferenceHistory
from models.inference_history_classification import InferenceHistoryClassification
from models.inference_history_people_tracking import InferenceHistoryPeopleTracking


class InferenceHistoryRepository:
    async def get_inference_history(self, db: AsyncSession)-> List[InferenceHistory]:
        result = await db.execute(select(InferenceHistory).order_by(InferenceHistory.created_at.desc()))
        return list(result.scalars().all())

    async def add_inference_history(self, db: AsyncSession, inference_history: InferenceHistory) -> InferenceHistory:
        db.add(inference_history)
        await db.commit()
        await db.refresh(inference_history)
        return inference_history

    async def add_inference_history_classification(self, db: AsyncSession, inference_history: InferenceHistoryClassification) -> InferenceHistoryClassification:
        db.add(inference_history)
        await db.commit()
        await db.refresh(inference_history)
        return inference_history

    async def add_inference_people_tracking(self, db: AsyncSession, inference_history: InferenceHistoryPeopleTracking) -> InferenceHistoryPeopleTracking:
        db.add(inference_history)
        await db.commit()
        await db.refresh(inference_history)
        return inference_history