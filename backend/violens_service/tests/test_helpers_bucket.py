import asyncio
import io
import os
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException

import helpers.bucket_helper as bucket_helper


def run(coro):
    return asyncio.run(coro)


class DummyBody:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    async def read(self, size: int) -> bytes:
        return self._stream.read(size)


class DummyPaginator:
    def __init__(self, pages):
        self._pages = pages

    async def paginate(self, **_kwargs):
        for page in self._pages:
            yield page


class DummyS3Client:
    def __init__(self) -> None:
        self.presigned_calls = []
        self.put_calls = []
        self.delete_calls = []
        self.delete_objects_calls = []
        self.paginator_pages = []
        self.get_object_payload = b"data"

    async def generate_presigned_url(self, **kwargs):
        self.presigned_calls.append(kwargs)
        return "http://signed-url"

    async def get_object(self, **_kwargs):
        return {"Body": DummyBody(self.get_object_payload)}

    async def put_object(self, **kwargs):
        self.put_calls.append(kwargs)

    async def delete_object(self, **kwargs):
        self.delete_calls.append(kwargs)

    async def delete_objects(self, **kwargs):
        self.delete_objects_calls.append(kwargs)

    def get_paginator(self, _name: str):
        return DummyPaginator(self.paginator_pages)


class DummyUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str = "application/octet-stream") -> None:
        self.filename = filename
        self.content_type = content_type
        self._stream = io.BytesIO(content)

    async def read(self, size: int | None = None) -> bytes:
        if size is None:
            return self._stream.read()
        return self._stream.read(size)

    async def seek(self, offset: int) -> None:
        self._stream.seek(offset)

    async def close(self) -> None:
        return None


def test_get_presigned_url(monkeypatch):
    client = DummyS3Client()

    @asynccontextmanager
    async def fake_client():
        yield client

    monkeypatch.setattr(bucket_helper, "_s3_client", fake_client)
    monkeypatch.setattr(bucket_helper, "BUCKET_NAME_DATASETS", "datasets")

    url = run(bucket_helper.get_presigned_url("path/file.mp4", expiration=120))

    assert url == "http://signed-url"
    assert client.presigned_calls[0]["Params"] == {"Bucket": "datasets", "Key": "path/file.mp4"}


def test_download_object_to_file(tmp_path, monkeypatch):
    client = DummyS3Client()
    client.get_object_payload = b"abc123"

    @asynccontextmanager
    async def fake_client():
        yield client

    monkeypatch.setattr(bucket_helper, "_s3_client", fake_client)
    monkeypatch.setattr(bucket_helper, "BUCKET_NAME_DATASETS", "datasets")

    target_path = tmp_path / "out.bin"
    result = run(bucket_helper.download_object_to_file("obj", str(target_path), chunk_size=2))

    assert result is True
    assert target_path.read_bytes() == b"abc123"


def test_create_unofficial_dataset_bucket_rejects_non_mp4():
    video = DummyUploadFile("video.avi", b"data", content_type="video/x-msvideo")

    with pytest.raises(HTTPException) as exc:
        run(bucket_helper.create_unofficial_dataset_bucket("dataset", [video]))

    assert exc.value.status_code == 400


def test_create_unofficial_dataset_bucket_uploads(monkeypatch):
    uploads = []

    async def fake_put_object(object_key: str, body: bytes, content_type: str):
        uploads.append((object_key, body, content_type))

    monkeypatch.setattr(bucket_helper, "put_object", fake_put_object)

    video1 = DummyUploadFile("video1.mp4", b"one", content_type="video/mp4")
    video2 = DummyUploadFile("video2.mp4", b"two", content_type="video/mp4")

    run(bucket_helper.create_unofficial_dataset_bucket("dataset", [video1, video2]))

    assert uploads == [
        ("dataset/video1.mp4", b"one", "video/mp4"),
        ("dataset/video2.mp4", b"two", "video/mp4"),
    ]


def test_upload_inference_model_success(monkeypatch, tmp_path):
    client = DummyS3Client()

    @asynccontextmanager
    async def fake_client():
        yield client

    monkeypatch.setattr(bucket_helper, "_s3_client", fake_client)
    monkeypatch.setattr(bucket_helper, "BUCKET_NAME_MODELS", "models")
    monkeypatch.setattr(bucket_helper, "validate_3dcnn_onnx", lambda _path: None)

    model_file = DummyUploadFile("model.onnx", b"content", content_type="application/octet-stream")
    key = run(bucket_helper.upload_inference_model("dataset", model_file))

    assert key == "models/dataset/model.onnx"
    assert client.put_calls[0]["Bucket"] == "models"


def test_upload_inference_model_too_large(monkeypatch):
    monkeypatch.setattr(bucket_helper, "MAX_MODEL_SIZE_BYTES", 4)
    monkeypatch.setattr(bucket_helper, "validate_3dcnn_onnx", lambda _path: None)
    monkeypatch.setattr(bucket_helper, "BUCKET_NAME_MODELS", "models")

    model_file = DummyUploadFile("model.onnx", b"abcdef", content_type="application/octet-stream")

    with pytest.raises(HTTPException) as exc:
        run(bucket_helper.upload_inference_model("dataset", model_file))

    assert exc.value.status_code == 413


def test_delete_inference_model_object_missing_bucket(monkeypatch):
    monkeypatch.setattr(bucket_helper, "BUCKET_NAME_MODELS", None)

    with pytest.raises(HTTPException) as exc:
        run(bucket_helper.delete_inference_model_object("models/dataset/model.onnx"))

    assert exc.value.status_code == 500


def test_get_used_storage_gb(monkeypatch):
    client = DummyS3Client()
    client.paginator_pages = [
        {"Contents": [{"Size": 1024}, {"Size": 2048}]},
        {"Contents": [{"Size": 1024}]},
    ]

    @asynccontextmanager
    async def fake_client():
        yield client

    monkeypatch.setattr(bucket_helper, "_s3_client", fake_client)
    monkeypatch.setattr(bucket_helper, "BUCKET_NAME_DATASETS", "datasets")

    used_gb = run(bucket_helper.get_used_storage_gb())

    expected = (1024 + 2048 + 1024) / (1024 ** 3)
    assert used_gb == expected

