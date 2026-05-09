from typing import List

from sqlalchemy import select, Sequence
from sqlalchemy.ext.asyncio import AsyncSession

from models.action import Action
from models.inference_action import InferenceAction


class InferenceActionsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_inference_actions(self) -> List[InferenceAction]:
        result = await self.db.execute(select(InferenceAction).order_by(InferenceAction.id))
        return list(result.scalars().all())

    async def get_inference_action_by_action_id(self, action_id: Action) -> InferenceAction | None:
        result = await self.db.execute(select(InferenceAction).filter(InferenceAction.action_id == action_id))
        return result.scalars().first()

    async def update_inference_action(self, inference_action: InferenceAction) -> InferenceAction:
        try:
            self.db.add(inference_action)
            await self.db.commit()
            await self.db.refresh(inference_action)
            return inference_action
        except Exception:
            await self.db.rollback()
            raise

    async def get_inference_action_by_id(self, id: int) -> InferenceAction | None:
        result = await self.db.execute(select(InferenceAction).filter(InferenceAction.id == id))
        return result.scalars().first()