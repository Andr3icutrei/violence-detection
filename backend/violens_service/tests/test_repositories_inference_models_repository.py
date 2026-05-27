import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from repositories.inference_models_repository import InferenceModelsRepository


def run(coro):
    return asyncio.run(coro)


class DummyScalars:
    def __init__(self, items):
        self._items = items

    def first(self):
        return self._items[0] if self._items else None


class DummyResult:
    def __init__(self, items=None, scalar_value=None):
        self._items = items or []
        self._scalar_value = scalar_value

    def scalars(self):
        return DummyScalars(self._items)

    def scalar_one(self):
        return self._scalar_value


def test_get_by_id_and_create_and_delete():
    model = SimpleNamespace(id=1)
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult([model]))
    db.commit = AsyncMock(return_value=None)
    db.refresh = AsyncMock(return_value=None)
    db.delete = AsyncMock(return_value=None)
    db.flush = AsyncMock(return_value=None)
    repo = InferenceModelsRepository(db)

    assert run(repo.get_by_id(1)) == model
    created = run(repo.create("name", "/path"))
    assert created.name == "name"
    run(repo.delete(created))


def test_count_datasets():
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult(scalar_value=5))
    repo = InferenceModelsRepository(db)

    assert run(repo.count_datasets(1)) == 5

