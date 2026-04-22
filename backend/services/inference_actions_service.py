from typing import Sequence, List

from sqlalchemy.ext.asyncio import AsyncSession

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