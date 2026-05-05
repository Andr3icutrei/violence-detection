from fastapi import Depends

from api.dependencies.datasets_repository import get_datasets_repository
from api.dependencies.inference_actions_repository import get_inference_actions_repository
from repositories.datasets_repository import DatasetsRepository
from repositories.inference_actions_repository import InferenceActionsRepository
from services.inference_actions_service import InferenceActionsService


def get_inference_actions_service(
    inference_actions_repository: InferenceActionsRepository = Depends(get_inference_actions_repository),
    datasets_repository: DatasetsRepository = Depends(get_datasets_repository),
) -> InferenceActionsService:
    return InferenceActionsService(
        inference_actions_repository=inference_actions_repository,
        dataset_repository=datasets_repository,
    )

