import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from repositories.inference_actions_repository import InferenceActionsRepository


def run(coro):
    return asyncio.run(coro)


class DummyScalars:
    def __init__(self, items):
        self._items = items

    def all(self):
        return list(self._items)

    def first(self):
        return self._items[0] if self._items else None


class DummyResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return DummyScalars(self._items)


def test_get_inference_actions():
    items = [SimpleNamespace(id=1)]
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult(items))
    repo = InferenceActionsRepository(db)

    assert run(repo.get_inference_actions()) == items


def test_get_inference_action_by_action_id():
    item = SimpleNamespace(id=1)
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult([item]))
    repo = InferenceActionsRepository(db)

    assert run(repo.get_inference_action_by_action_id(1)) == item


def test_update_inference_action_success():
    item = SimpleNamespace(id=1)
    db = Mock()
    db.commit = AsyncMock(return_value=None)
    db.refresh = AsyncMock(return_value=None)
    repo = InferenceActionsRepository(db)

    result = run(repo.update_inference_action(item))

    assert result == item


def test_update_inference_action_rollback_on_error():
    item = SimpleNamespace(id=1)
    db = Mock()
    db.commit = AsyncMock(side_effect=SQLAlchemyError("boom"))
    db.refresh = AsyncMock(return_value=None)
    db.rollback = AsyncMock(return_value=None)
    repo = InferenceActionsRepository(db)

    with pytest.raises(SQLAlchemyError):
        run(repo.update_inference_action(item))

    db.rollback.assert_awaited_once()

