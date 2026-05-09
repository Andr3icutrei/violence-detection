from typing import List

from sqlalchemy import select, extract
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from models import InferenceHistory
from models.inference_history_classification import InferenceHistoryClassification
from models.inference_history_people_tracking import InferenceHistoryPeopleTracking


class InferenceHistoryRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_inference_history(self)-> List[InferenceHistory]:
        result = await self.db.execute(select(InferenceHistory).order_by(InferenceHistory.created_at.desc()))
        return list(result.scalars().all())

    async def add_inference_history(self, inference_history: InferenceHistory) -> InferenceHistory:
        self.db.add(inference_history)
        await self.db.commit()
        await self.db.refresh(inference_history)
        return inference_history

    async def add_inference_history_classification(self, inference_history: InferenceHistoryClassification) -> InferenceHistoryClassification:
        self.db.add(inference_history)
        await self.db.commit()
        await self.db.refresh(inference_history)
        return inference_history

    async def add_inference_people_tracking(self, inference_history: InferenceHistoryPeopleTracking) -> InferenceHistoryPeopleTracking:
        self.db.add(inference_history)
        await self.db.commit()
        await self.db.refresh(inference_history)
        return inference_history

    async def get_classification_inference_history(self, year: int, month: int) -> List[InferenceHistoryClassification]:
        result = await self.db.execute(
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

    async def get_people_tracking_inference_history(self, year: int, month: int) -> List[InferenceHistoryPeopleTracking]:
        result = await self.db.execute(
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