from datetime import datetime
from typing import List

from pydantic import BaseModel


class InferenceHistoryClassificationStatsResponseDto(BaseModel):
    id: int
    ground_truth: bool
    prediction: bool
    created_at: datetime

class InferenceHistoryPeopleTrackingStatsResponseDto(BaseModel):
    id: int
    people_tracked: int
    created_at: datetime

class InferenceHistoryStatsResponseDto(BaseModel):
    classification_runs: List[InferenceHistoryClassificationStatsResponseDto]
    people_tracking_runs: List[InferenceHistoryPeopleTrackingStatsResponseDto]