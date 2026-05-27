import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from schemas.inference_actions_schema import PatchInferenceActionRequestDto
from services.inference_actions_service import InferenceActionsService


def run(coro):
    return asyncio.run(coro)


def test_get_inference_actions_for_dataset_missing_dataset():
    dataset_repo = Mock()
    dataset_repo.get_by_id = AsyncMock(return_value=None)
    actions_repo = Mock()
    actions_repo.get_inference_actions = AsyncMock(return_value=[])
    service = InferenceActionsService(actions_repo, dataset_repo)

    with pytest.raises(HTTPException) as exc:
        run(service.get_inference_actions_for_dataset(1))

    assert exc.value.status_code == 404


def test_update_credits_for_action_success():
    dataset_repo = Mock()
    actions_repo = Mock()
    action_record = SimpleNamespace(id=1, credits=5)
    actions_repo.get_inference_action_by_id = AsyncMock(return_value=action_record)
    actions_repo.update_inference_action = AsyncMock(return_value=None)
    service = InferenceActionsService(actions_repo, dataset_repo)

    payload = PatchInferenceActionRequestDto(actions=[{"id": 1, "new_credits": 9}])

    run(service.update_credits_for_action(payload))

    assert action_record.credits == 9
    actions_repo.update_inference_action.assert_awaited_once_with(action_record)


def test_update_credits_for_action_missing_action_returns_500():
    dataset_repo = Mock()
    actions_repo = Mock()
    actions_repo.get_inference_action_by_id = AsyncMock(return_value=None)
    service = InferenceActionsService(actions_repo, dataset_repo)

    payload = PatchInferenceActionRequestDto(actions=[{"id": 1, "new_credits": 9}])

    with pytest.raises(HTTPException) as exc:
        run(service.update_credits_for_action(payload))

    assert exc.value.status_code == 500


def test_get_inference_actions_stats_error_returns_500():
    dataset_repo = Mock()
    actions_repo = Mock()
    actions_repo.get_inference_actions = AsyncMock(side_effect=RuntimeError("boom"))
    service = InferenceActionsService(actions_repo, dataset_repo)

    with pytest.raises(HTTPException) as exc:
        run(service.get_inference_actions_stats())

    assert exc.value.status_code == 500

