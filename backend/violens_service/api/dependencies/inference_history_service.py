from fastapi import Depends

from api.dependencies.inference_history_repository import get_inference_history_repository
from repositories.inference_history_repository import InferenceHistoryRepository
from services.inference_history_service import InferenceHistoryService


def get_inference_history_service(
    inference_history_repository: InferenceHistoryRepository = Depends(get_inference_history_repository),
) -> InferenceHistoryService:
    return InferenceHistoryService(
        inference_history_repository=inference_history_repository,
    )

