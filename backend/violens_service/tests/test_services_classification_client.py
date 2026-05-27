import asyncio

import pytest
from fastapi import HTTPException

import services.classification_client as classification_client
from services.classification_client import ClassificationClient


def run(coro):
    return asyncio.run(coro)


class DummyResponse:
    def __init__(self, status_code: int, json_data: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self) -> dict:
        return self._json_data


class DummyAsyncClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = False
        self.response = DummyResponse(200, {"ok": True})

    async def get(self, url: str, params: dict, timeout: float):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def test_classification_client_success(monkeypatch):
    dummy_client = DummyAsyncClient()
    monkeypatch.setattr(classification_client.httpx, "AsyncClient", lambda **_kwargs: dummy_client)

    async def _run():
        async with ClassificationClient("http://host/", 1.5, verify_ssl=True) as client:
            return await client.classify_video("video.mp4", "model.onnx")

    result = run(_run())

    assert result == {"ok": True}
    assert dummy_client.calls[0]["url"] == "http://host/classification/classify_video"
    assert dummy_client.calls[0]["params"] == {"video_path": "video.mp4", "inference_model_path": "model.onnx"}
    assert dummy_client.closed is True


def test_classification_client_non_200_raises(monkeypatch):
    dummy_client = DummyAsyncClient()
    dummy_client.response = DummyResponse(500, {"error": "bad"}, text="bad")
    monkeypatch.setattr(classification_client.httpx, "AsyncClient", lambda **_kwargs: dummy_client)

    async def _run():
        async with ClassificationClient("http://host", 1.5) as client:
            await client.classify_video("video.mp4", "model.onnx")

    with pytest.raises(HTTPException) as exc:
        run(_run())

    assert exc.value.status_code == 500


def test_classification_client_requires_context_manager():
    async def _run():
        client = ClassificationClient("http://host", 1.5)
        await client.classify_video("video.mp4", "model.onnx")

    with pytest.raises(RuntimeError):
        run(_run())

