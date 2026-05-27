import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

import services.datasets_service as datasets_service
from models.dataset_status import DatasetStatus
from schemas.videos_schema import ReviewVideoRequestDto
from services.datasets_service import DatasetsService
from services.model_validation import ConfusionMatrixCounts


def run(coro):
    return asyncio.run(coro)


def make_service():
    datasets_repo = Mock()
    users_repo = Mock()
    videos_repo = Mock()
    models_repo = Mock()
    notifier = Mock()
    notifier.broadcast_dataset_updated = AsyncMock(return_value=None)
    return DatasetsService(datasets_repo, users_repo, videos_repo, models_repo, notifier)


def test_ensure_dataset_name_available_conflict():
    service = make_service()
    service.datasets_repository.get_by_name = AsyncMock(return_value=SimpleNamespace(id=1))

    with pytest.raises(HTTPException) as exc:
        run(service._ensure_dataset_name_available("name"))

    assert exc.value.status_code == 409


def test_get_user_or_404():
    service = make_service()
    service.users_repository.get_by_id = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as exc:
        run(service._get_user_or_404(1))

    assert exc.value.status_code == 404


def test_ensure_user_has_no_pending_datasets_conflict():
    service = make_service()
    service.datasets_repository.user_has_pending_datasets = AsyncMock(return_value=True)

    with pytest.raises(HTTPException) as exc:
        run(service._ensure_user_has_no_pending_datasets(1))

    assert exc.value.status_code == 409


def test_ensure_user_has_no_pending_datasets_error():
    service = make_service()
    service.datasets_repository.user_has_pending_datasets = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(HTTPException) as exc:
        run(service._ensure_user_has_no_pending_datasets(1))

    assert exc.value.status_code == 500


def test_create_unofficial_dataset_record_cleans_up_on_error(monkeypatch):
    service = make_service()
    service.datasets_repository.create_unofficial_dataset = AsyncMock(side_effect=RuntimeError("fail"))
    cleanup_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(service, "_cleanup_inference_model_record", cleanup_mock)

    model_record = SimpleNamespace(id=2, path="models/path")

    async def fake_create_model(_dto):
        return model_record

    monkeypatch.setattr(service, "_create_inference_model_record", fake_create_model)

    create_dataset_dto = SimpleNamespace(
        name="dataset",
        inference_model=object(),
        inference_model_name="model",
        videos=[],
    )

    with pytest.raises(HTTPException) as exc:
        run(service._create_unofficial_dataset_record(create_dataset_dto, user_id=1, is_official=False))

    assert exc.value.status_code == 500
    cleanup_mock.assert_awaited_once_with(model_record)


def test_build_validation_inputs_no_videos_left():
    service = make_service()
    dataset = SimpleNamespace(videos=[SimpleNamespace(id=1), SimpleNamespace(id=2)])
    review = [ReviewVideoRequestDto(video_id=1, is_violent=True), ReviewVideoRequestDto(video_id=2, is_violent=False)]

    with pytest.raises(HTTPException) as exc:
        service._build_validation_inputs(dataset, review, excluded_video_ids=[1, 2])

    assert exc.value.status_code == 400


def test_get_inference_model_or_400_missing():
    service = make_service()
    service.inference_models_repository.get_by_id = AsyncMock(return_value=None)
    dataset = SimpleNamespace(inference_model_id=1)

    with pytest.raises(HTTPException) as exc:
        run(service._get_inference_model_or_400(dataset))

    assert exc.value.status_code == 400


def test_delete_inference_model_if_unassigned_deletes(monkeypatch):
    service = make_service()
    service.inference_models_repository.count_datasets = AsyncMock(return_value=0)
    model = SimpleNamespace(id=9, path="models/path")
    service.inference_models_repository.get_by_id = AsyncMock(return_value=model)
    service.inference_models_repository.delete = AsyncMock(return_value=None)
    delete_object = AsyncMock(return_value=None)
    monkeypatch.setattr(datasets_service, "delete_inference_model_object", delete_object)

    run(service._delete_inference_model_if_unassigned(9))

    delete_object.assert_awaited_once_with("models/path")
    service.inference_models_repository.delete.assert_awaited_once_with(model)


def test_build_validate_model_response():
    service = make_service()
    counts = ConfusionMatrixCounts(true_positive=3, true_negative=1, false_positive=1, false_negative=0)

    result = service._build_validate_model_response(counts)

    assert result.accuracy == 4 / 5
    assert result.confusion_matrix.true_positive == 3

