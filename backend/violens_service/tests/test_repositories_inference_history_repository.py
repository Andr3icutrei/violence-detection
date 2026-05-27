import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from repositories.inference_history_repository import InferenceHistoryRepository


def run(coro):
    return asyncio.run(coro)


class DummyScalars:
    def __init__(self, items):
        self._items = items

    def all(self):
        return list(self._items)


class DummyResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return DummyScalars(self._items)


def test_get_and_add_inference_history():
    history = SimpleNamespace(id=1)
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult([history]))
    db.commit = AsyncMock(return_value=None)
    db.refresh = AsyncMock(return_value=None)
    repo = InferenceHistoryRepository(db)

    assert run(repo.get_inference_history()) == [history]
    assert run(repo.add_inference_history(history)) == history


def test_add_inference_history_types():
    classification = SimpleNamespace(id=2)
    tracking = SimpleNamespace(id=3)
    db = Mock()
    db.commit = AsyncMock(return_value=None)
    db.refresh = AsyncMock(return_value=None)
    repo = InferenceHistoryRepository(db)

    assert run(repo.add_inference_history_classification(classification)) == classification
    assert run(repo.add_inference_people_tracking(tracking)) == tracking


def test_get_history_filters():
    records = [SimpleNamespace(id=1)]
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult(records))
    repo = InferenceHistoryRepository(db)

    assert run(repo.get_classification_inference_history(2024, 1)) == records
    assert run(repo.get_people_tracking_inference_history(2024, 1)) == records

