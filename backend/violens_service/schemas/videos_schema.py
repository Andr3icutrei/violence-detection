from typing import List

from pydantic import BaseModel, ConfigDict, Field

class VideoResponseDto(BaseModel):
    id: int
    uid: str
    name: str
    path: str
    dataset_id: int
    dataset_name: str
    dataset_is_official: bool
    is_violent: bool
    duration: int
    frame_rate: int

class InferenceVideoRequestDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    video_id: str = Field(alias="videoId")
    action_ids: List[int] = Field(alias="actionIds")


class InferenceClassificationGradcamResponseDto(BaseModel):
    gt_label: str
    predicted_label: str
    predicted_probability: float
    confidence: float
    gradcam_video_url: str


class PeopleTrackingResponseDto(BaseModel):
    tracked_video_url: str

class ReviewVideoRequestDto(BaseModel):
    video_id: int
    is_violent: bool
