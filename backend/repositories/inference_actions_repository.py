from typing import List

from sqlalchemy import select, Sequence
from sqlalchemy.ext.asyncio import AsyncSession

from models.action import Action
from models.inference_action import InferenceAction


class InferenceActionsRepository:
    async def get_inference_actions(self, db: AsyncSession) -> List[InferenceAction]:
        result = await db.execute(select(InferenceAction))
        return list(result.scalars().all())

    async def get_inference_action_by_action_id(self, action_id: Action, db: AsyncSession) -> InferenceAction | None:
        result = await db.execute(select(InferenceAction).filter(InferenceAction.action_id == action_id))
        return result.scalars().first()

    async def update_inference_action(self, inference_action: InferenceAction, db: AsyncSession) -> InferenceAction:
        try:
            db.add(inference_action)
            await db.commit()
            await db.refresh(inference_action)
            return inference_action
        except Exception:
            await db.rollback()
            raise