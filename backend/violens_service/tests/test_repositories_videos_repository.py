import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from repositories.videos_repository import VideosRepository


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
    def __init__(self, items=None):
        self._items = items or []

    def scalars(self):
        return DummyScalars(self._items)


def test_get_videos_paged_with_search_and_filters():
    videos = [SimpleNamespace(id=1)]
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult(videos))
    repo = VideosRepository(db)

    result = run(repo.get_videos_paged(
        search_term="non-violent",
        dataset_id=1,
        is_violent=False,
        dataset_status=None,
        asc=True,
        page=0,
        page_size=10,
    ))

    assert result == videos


def test_get_by_uid_invalid():
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult([]))
    repo = VideosRepository(db)

    assert run(repo.get_by_uid("not-a-uuid")) is None


def test_delete_removes_history_and_video():
    db = Mock()
    db.execute = AsyncMock(side_effect=[
        DummyResult([1, 2]),
        DummyResult([]),
        DummyResult([]),
        DummyResult([]),
    ])
    db.delete = AsyncMock(return_value=None)
    db.flush = AsyncMock(return_value=None)
    repo = VideosRepository(db)

    video = SimpleNamespace(id=1)

    run(repo.delete(video))

    db.delete.assert_awaited_once_with(video)


def test_save_video():
    db = Mock()
    db.commit = AsyncMock(return_value=None)
    db.refresh = AsyncMock(return_value=None)
    repo = VideosRepository(db)

    video = SimpleNamespace(id=1)
    result = run(repo.save(video))

    assert result == video

