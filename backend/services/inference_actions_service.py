from typing import Sequence, List

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from models import Dataset
from models.action import Action
from models.inference_action import InferenceAction
from repositories.datasets_repository import DatasetsRepository
from repositories.inference_actions_repository import InferenceActionsRepository
from schemas.inference_actions_schema import PatchInferenceActionRequestDto


class InferenceActionsService:
    def __init__(
        self,
        inference_actions_repository: InferenceActionsRepository,
        dataset_repository: DatasetsRepository
    ):
        self.inference_actions_repository = inference_actions_repository
        self.dataset_repository = dataset_repository

    async def get_inference_actions_for_dataset(self, dataset_id: int) -> List[InferenceAction]:
        result: List[InferenceAction] = await self.inference_actions_repository.get_inference_actions()
        dataset: Dataset = await self.dataset_repository.get_by_id(dataset_id)
        if not dataset.is_official:
            result = [action for action in result if action.action_id == Action.PEOPLE_TRACKING]
        return result

    async def update_credits_for_action(self, actions_to_patch: PatchInferenceActionRequestDto) -> None:
        try:
            for action in actions_to_patch.actions:
                inference_action_to_update: InferenceAction | None = await self.inference_actions_repository.get_inference_action_by_id(action.id)
                if not inference_action_to_update:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Inference action not found."
                    )
                inference_action_to_update.credits = action.new_credits
                await self.inference_actions_repository.update_inference_action(inference_action_to_update)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while updating the credits for the action."
            )

    async def get_inference_actions_stats(self) -> List[InferenceAction]:
        try:
            return await self.inference_actions_repository.get_inference_actions()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while retrieving inference actions statistics."
            )