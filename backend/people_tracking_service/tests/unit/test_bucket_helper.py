import asyncio

from helpers import bucket_helper


class DummyBody:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._index = 0

    async def read(self, _chunk_size):
        if self._index >= len(self._chunks):
            return b""
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class DummyClient:
    def __init__(self, should_fail=False, payload=b""):
        self._should_fail = should_fail
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get_object(self, Bucket, Key):
        if self._should_fail:
            raise RuntimeError("not found")
        return {"Body": DummyBody([self._payload])}


class DummySession:
    def __init__(self, client):
        self._client = client

    def client(self, **kwargs):
        return self._client


def run_async(coro):
    return asyncio.run(coro)


def test_download_object_to_file_success(monkeypatch, tmp_path):
    target = tmp_path / "video.mp4"
    dummy_client = DummyClient(payload=b"data")
    monkeypatch.setattr(bucket_helper, "session", DummySession(dummy_client))

    was_downloaded = run_async(bucket_helper.download_object_to_file("key", str(target)))

    assert was_downloaded is True
    assert target.read_bytes() == b"data"


def test_download_object_to_file_not_found(monkeypatch, tmp_path):
    target = tmp_path / "video.mp4"
    dummy_client = DummyClient(should_fail=True)
    monkeypatch.setattr(bucket_helper, "session", DummySession(dummy_client))

    was_downloaded = run_async(bucket_helper.download_object_to_file("missing", str(target)))

    assert was_downloaded is False
    assert not target.exists()
