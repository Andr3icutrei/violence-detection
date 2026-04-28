from typing import List

from sqlalchemy import select, extract
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from models import InferenceHistory
from models.inference_history_classification import InferenceHistoryClassification
from models.inference_history_people_tracking import InferenceHistoryPeopleTracking


class InferenceHistoryRepository:
    async def get_inference_history(self, db: AsyncSession)-> List[InferenceHistory]:
        result = await db.execute(select(InferenceHistory).order_by(InferenceHistory.created_at.desc()))
        return list(result.scalars().all())

    async def add_inference_history(self, inference_history: InferenceHistory, db: AsyncSession) -> InferenceHistory:
        db.add(inference_history)
        await db.commit()
        await db.refresh(inference_history)
        return inference_history

    async def add_inference_history_classification(self, inference_history: InferenceHistoryClassification, db: AsyncSession) -> InferenceHistoryClassification:
        db.add(inference_history)
        await db.commit()
        await db.refresh(inference_history)
        return inference_history

    async def add_inference_people_tracking(self, inference_history: InferenceHistoryPeopleTracking, db: AsyncSession) -> InferenceHistoryPeopleTracking:
        db.add(inference_history)
        await db.commit()
        await db.refresh(inference_history)
        return inference_history

    async def get_classification_inference_history(self, year: int, month: int, db: AsyncSession) -> List[InferenceHistoryClassification]:
        result = await db.execute(
            select(InferenceHistoryClassification)
            .options(joinedload(InferenceHistoryClassification.inference_history))
            .join(InferenceHistoryClassification.inference_history)
            .where(
                extract('year', InferenceHistory.created_at) == year,
                extract('month', InferenceHistory.created_at) == month
            )
            .order_by(InferenceHistory.created_at.asc())
        )
        return list(result.scalars().all())

    async def get_people_tracking_inference_history(self, year: int, month: int, db: AsyncSession) -> List[InferenceHistoryPeopleTracking]:
        result = await db.execute(
            select(InferenceHistoryPeopleTracking)
            .options(joinedload(InferenceHistoryPeopleTracking.inference_history))
            .join(InferenceHistoryPeopleTracking.inference_history)
            .where(
                extract('year', InferenceHistory.created_at) == year,
                extract('month', InferenceHistory.created_at) == month
            )
            .order_by(InferenceHistory.created_at.asc())
        )
        return list(result.scalars().all())