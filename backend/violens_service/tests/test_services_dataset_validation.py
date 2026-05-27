from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from schemas.videos_schema import ReviewVideoRequestDto
from services.dataset_validation import (
    build_review_video_map,
    ensure_all_videos_reviewed,
    get_review_video_or_raise,
    validate_excluded_ids,
    validate_review_ids,
)


def test_build_review_video_map():
    videos = [ReviewVideoRequestDto(video_id=1, is_violent=True), ReviewVideoRequestDto(video_id=2, is_violent=False)]
    result = build_review_video_map(videos)
    assert set(result.keys()) == {1, 2}


def test_validate_excluded_ids_raises_for_missing():
    with pytest.raises(HTTPException) as exc:
        validate_excluded_ids({1, 2}, {2, 3})
    assert exc.value.status_code == 400
    assert "Excluded video ids" in exc.value.detail


def test_validate_review_ids_raises_for_missing():
    with pytest.raises(HTTPException) as exc:
        validate_review_ids({1, 2}, {1, 3})
    assert exc.value.status_code == 400
    assert "Video ids not found" in exc.value.detail


def test_get_review_video_or_raise_missing():
    with pytest.raises(HTTPException) as exc:
        get_review_video_or_raise({}, 5, "review")
    assert exc.value.status_code == 400


def test_ensure_all_videos_reviewed_missing():
    dataset_videos = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    with pytest.raises(HTTPException) as exc:
        ensure_all_videos_reviewed(dataset_videos, {1}, "review")
    assert exc.value.status_code == 400

