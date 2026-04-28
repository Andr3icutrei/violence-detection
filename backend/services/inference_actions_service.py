from typing import Sequence, List

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from models import Dataset
from models.action import Action
from models.inference_action import InferenceAction
from repositories.datasets_repository import DatasetsRepository
from repositories.inference_actions_repository import InferenceActionsRepository


class InferenceActionsService:
    def __init__(self):
        self.inference_actions_repository = InferenceActionsRepository()
        self.dataset_repository = DatasetsRepository()

    async def get_inference_actions_for_dataset(self, dataset_id: int, db: AsyncSession) -> List[InferenceAction]:
        result: List[InferenceAction] = await self.inference_actions_repository.get_inference_actions(db)
        dataset: Dataset = await self.dataset_repository.get_by_id(dataset_id, db)
        if not dataset.is_official:
            result = [action for action in result if action.action_id == Action.PEOPLE_TRACKING]
        return result

    async def update_credits_for_action(self, action_id: Action, credits: int, db: AsyncSession) -> InferenceAction:
        try:
            inference_action_to_update: InferenceAction | None = await self.inference_actions_repository.get_inference_action_by_action_id(action_id, db)
            if not inference_action_to_update:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Inference action not found."
                )
            inference_action_to_update.credits = credits
            return await self.inference_actions_repository.update_inference_action(inference_action_to_update, db)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while updating the credits for the action."
            )

    async def get_inference_actions_stats(self, db: AsyncSession) -> List[InferenceAction]:
        try:
            return await self.inference_actions_repository.get_inference_actions(db)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while retrieving inference actions statistics."
            )