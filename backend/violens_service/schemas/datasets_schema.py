from dataclasses import dataclass
from typing import List, Annotated, Protocol

from fastapi import UploadFile, Form, File
from pydantic import BaseModel

from models.dataset_status import DatasetStatus
from schemas.users_schema import UserResponseDto
from schemas.videos_schema import VideoResponseDto, ReviewVideoRequestDto


class DatasetResponseDto(BaseModel):
    id: int
    name: str
    is_official: bool
    status: DatasetStatus

class DatasetToReviewResponseDto(DatasetResponseDto):
    user: UserResponseDto
    videos_count: int = 0
    violent_videos_count: int = 0
    non_violent_videos_count: int = 0

@dataclass
class CreateDatasetRequestDto:
    name: str
    videos: list[UploadFile]

    @classmethod
    def as_form(
        cls,
        name: Annotated[str, Form(...)],
        videos: Annotated[list[UploadFile], File(...)],
    ) -> "CreateDatasetRequestDto":
        return cls(name=name, videos=videos)

class DatasetWithVideosResponseDto(DatasetResponseDto):
    videos: List[VideoResponseDto]

class ReviewDatasetRequestDto(BaseModel):
    is_approved: bool
    videos: List[ReviewVideoRequestDto]
    review_comment: str

class EditDatasetRequestDto(BaseModel):
    videos: List[ReviewVideoRequestDto]

class MostPopularDatasetResponseDto(DatasetResponseDto):
    inferences_videos_count: int

class DatasetsStatsResponseDto(BaseModel):
    most_popular_dataset_classification: MostPopularDatasetResponseDto
    most_popular_dataset_people_tracking: MostPopularDatasetResponseDto
    official_datasets_count: int
    unofficial_datasets_count: int
    pending_datasets_count: int
    storage_used_gb: float
