from typing import List

from models.inference_history_classification import InferenceHistoryClassification
from models.inference_history_people_tracking import InferenceHistoryPeopleTracking
from repositories.inference_history_repository import InferenceHistoryRepository
from schemas.inference_history_schema import InferenceHistoryStatsResponseDto, \
    InferenceHistoryClassificationStatsResponseDto, InferenceHistoryPeopleTrackingStatsResponseDto


class InferenceHistoryService:
    def __init__(
        self,
        inference_history_repository: InferenceHistoryRepository
    ):
        self.inference_history_repository: InferenceHistoryRepository = inference_history_repository

    async def get_inference_history_stats(self, year: int, month: int) -> InferenceHistoryStatsResponseDto:
        classification_runs: List[InferenceHistoryClassification] = await self.inference_history_repository.get_classification_inference_history(year, month)
        people_tracking_runs: List[InferenceHistoryPeopleTracking] = await self.inference_history_repository.get_people_tracking_inference_history(year, month)

        return InferenceHistoryStatsResponseDto(
            classification_runs=[
                InferenceHistoryClassificationStatsResponseDto(
                    id=run.id,
                    prediction=bool(run.prediction),
                    ground_truth=bool(run.ground_truth),
                    created_at=run.inference_history.created_at
                ) for run in classification_runs
            ],
            people_tracking_runs=[
                InferenceHistoryPeopleTrackingStatsResponseDto(
                    id=run.id,
                    people_tracked=run.people_tracked,
                    created_at=run.inference_history.created_at
                ) for run in people_tracking_runs
            ]
        )