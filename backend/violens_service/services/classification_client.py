import httpx
from fastapi import HTTPException
from starlette import status


class ClassificationClient:
    def __init__(self, base_url: str, timeout_seconds: float, verify_ssl: bool = False) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._verify_ssl = verify_ssl
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ClassificationClient":
        self._client = httpx.AsyncClient(verify=self._verify_ssl, timeout=self._timeout_seconds)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def classify_video(self, video_path: str, inference_model_path: str) -> dict:
        if self._client is None:
            raise RuntimeError("ClassificationClient is not initialized. Use 'async with'.")
        response = await self._client.get(
            f"{self._base_url}/classification/classify_video",
            params={"video_path": video_path, "inference_model_path": inference_model_path},
            timeout=self._timeout_seconds,
        )
        if response.status_code != status.HTTP_200_OK:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()

