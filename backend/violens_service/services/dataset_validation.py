from typing import Iterable

from fastapi import HTTPException
from starlette import status

from schemas.videos_schema import ReviewVideoRequestDto


def build_review_video_map(videos: list[ReviewVideoRequestDto]) -> dict[int, ReviewVideoRequestDto]:
    return {video.video_id: video for video in videos}


def validate_excluded_ids(dataset_video_ids: set[int], excluded_ids: set[int]) -> None:
    missing_excluded = excluded_ids - dataset_video_ids
    if missing_excluded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Excluded video ids not found in dataset: {sorted(missing_excluded)}",
        )


def validate_review_ids(dataset_video_ids: set[int], review_video_ids: set[int]) -> None:
    invalid_review_ids = review_video_ids - dataset_video_ids
    if invalid_review_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video ids not found in dataset: {sorted(invalid_review_ids)}",
        )


def get_review_video_or_raise(
    review_video_map: dict[int, ReviewVideoRequestDto],
    video_id: int,
    context: str,
) -> ReviewVideoRequestDto:
    review_video_dto = review_video_map.get(video_id)
    if review_video_dto is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video with id {video_id} is missing in the {context} request.",
        )
    return review_video_dto


def ensure_all_videos_reviewed(
    dataset_videos: Iterable,
    review_video_ids: set[int],
    context: str,
) -> None:
    for video in dataset_videos:
        if video.id not in review_video_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video with id {video.id} is missing in the {context} request.",
            )

