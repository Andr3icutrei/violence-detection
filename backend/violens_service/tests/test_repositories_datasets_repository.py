import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from models.dataset_status import DatasetStatus
from repositories.datasets_repository import DatasetsRepository


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
    def __init__(self, scalars_items=None, scalar_value=None, first_value=None, all_value=None):
        self._scalars_items = scalars_items or []
        self._scalar_value = scalar_value
        self._first_value = first_value
        self._all_value = all_value

    def scalars(self):
        return DummyScalars(self._scalars_items if self._all_value is None else self._all_value)

    def scalar_one(self):
        return self._scalar_value

    def first(self):
        return self._first_value

    def all(self):
        return self._all_value or []


def test_get_by_name_and_id():
    dataset = SimpleNamespace(id=1, name="ds")
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult([dataset]))
    repo = DatasetsRepository(db)

    assert run(repo.get_by_name("ds")) == dataset
    assert run(repo.get_by_id(1)) == dataset


def test_get_all_with_filters():
    dataset = SimpleNamespace(id=1, name="ds")
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult([dataset]))
    repo = DatasetsRepository(db)

    result = run(repo.get_all("term", page=1, page_size=5, dataset_status=DatasetStatus.PENDING, is_official=True))

    assert result == [dataset]


def test_create_unofficial_dataset_creates_records(monkeypatch, tmp_path):
    db = Mock()
    db.add = Mock()
    db.commit = AsyncMock(return_value=None)
    db.refresh = AsyncMock(return_value=None)
    repo = DatasetsRepository(db)

    class DummyCapture:
        def get(self, prop):
            return 30.0 if prop == 5 else 60

        def release(self):
            return None

    monkeypatch.setattr("repositories.datasets_repository.cv2.VideoCapture", lambda _path: DummyCapture())

    class DummyUploadFile:
        def __init__(self, filename, data):
            self.filename = filename
            self._data = data

        async def seek(self, _offset):
            return None

        async def read(self):
            return self._data

    videos = [DummyUploadFile("video.mp4", b"data")]

    dataset = run(repo.create_unofficial_dataset("dataset", videos, user_id=1, inference_model_id=2, is_official=False))

    assert dataset.name == "dataset"
    assert len(dataset.videos) == 1


def test_delete_sets_deleted_at():
    dataset = SimpleNamespace(deleted_at=None)
    db = Mock()
    db.add = Mock()
    db.commit = AsyncMock(return_value=None)
    db.refresh = AsyncMock(return_value=None)
    repo = DatasetsRepository(db)

    run(repo.delete(dataset))

    assert dataset.deleted_at is not None


def test_counts_and_popular_queries():
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult(scalar_value=3, first_value=("dataset", 5)))
    repo = DatasetsRepository(db)

    assert run(repo.get_official_datasets_count()) == 3
    assert run(repo.get_unofficial_datasets_count()) == 3
    assert run(repo.get_pending_datasets_count()) == 3
    assert run(repo.get_most_popular_dataset_classification()) == ("dataset", 5)
    assert run(repo.get_most_popular_dataset_people_tracking()) == ("dataset", 5)

