from typing import Sequence, List

from sqlalchemy.ext.asyncio import AsyncSession

from models.inference_action import InferenceAction
from repositories.inference_actions_repository import InferenceActionsRepository


class InferenceActionsService:
    def __init__(self):
        self.inference_actions_repository = InferenceActionsRepository()

    async def get_inference_actions(self, db: AsyncSession) -> List[InferenceAction]:
        result: Sequence[InferenceAction] = await self.inference_actions_repository.get_inference_actions(db)
        return list(result)