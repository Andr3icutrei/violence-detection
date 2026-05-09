import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import List, Sequence

import httpx
from fastapi import HTTPException
from shared_models import InferenceModel
from starlette import status

from helpers.bucket_helper import get_presigned_url
from helpers.env_helper import get_env_variable

from models import Video, InferenceHistory, User
from models.action import Action
from models.inference_action import InferenceAction
from models.inference_history_classification import InferenceHistoryClassification
from models.inference_history_people_tracking import InferenceHistoryPeopleTracking
from repositories.inference_actions_repository import InferenceActionsRepository

from repositories.inference_history_repository import InferenceHistoryRepository
from repositories.users_repository import UsersRepository
from repositories.videos_repository import VideosRepository

@dataclass
class InferenceVideoResult:
    video_path: str
    predicted_label: str
    confidence: float
    predicted_class_probability: float

class VideosService:
    def __init__(
        self,
        videos_repository: VideosRepository,
        inference_history_repository: InferenceHistoryRepository,
        users_repository: UsersRepository,
        inference_actions_repository: InferenceActionsRepository,
    ):
        self.videos_repository = videos_repository
        self.inference_history_repository = inference_history_repository
        self.users_repository = users_repository
        self.inference_actions_repository = inference_actions_repository

    async def get_videos_paged(
        self,
        search_term: str | None,
        dataset_id: int | None,
        is_violent: bool | None = None,
        asc: bool = True,
        page: int = 0,
        page_size: int = 40,
    ) -> List[Video]:
        videos: Sequence[Video] = await (
            self.videos_repository.get_videos_paged(search_term, dataset_id, is_violent, asc, page, page_size))

        tasks = [get_presigned_url(video.path) for video in videos]
        presigned_urls = await asyncio.gather(*tasks)

        result = []
        for video, url in zip(videos, presigned_urls):
            video.path = url
            result.append(video)

        return result

    async def exists_video(self, video_uid: str) -> bool:
        video: Video = await self.videos_repository.get_by_uid(video_uid)
        return video is not None

    @staticmethod
    def _to_label(is_violent: bool) -> str:
        return "violent" if is_violent else "non-violent"

    async def _get_user_or_404(self, user_id: int) -> User:
        user = await self.users_repository.get_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found."
            )
        return user

    async def _get_video_or_404(self, video_id: int, *, for_classification: bool = False) -> Video:
        video = await (
            self.videos_repository.get_by_id_for_classification(video_id)
            if for_classification
            else self.videos_repository.get_by_id(video_id)
        )
        if video is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found."
            )
        return video

    async def _ensure_credits(self, user: User, action: InferenceAction) -> None:
        if user.credits - action.credits < 0:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Not enough credits to perform this action."
            )

    async def _deduct_credits(self, user: User, action: InferenceAction) -> None:
        user.credits = user.credits - action.credits
        try:
            await self.users_repository.add_user(user)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error while updating user credits: {str(e)}"
            ) from e

    async def classify_and_gradcam_video(self, video_id: int, current_user: User) -> InferenceVideoResult:
        overlay_video_path: str | None = None
        should_cleanup = True
        db_user = await self._get_user_or_404(current_user.id)
        inference_action = await self.inference_actions_repository.get_inference_action_by_action_id(
            Action.CLASSIFICATION
        )
        await self._ensure_credits(db_user, inference_action)
        video = await self._get_video_or_404(video_id, for_classification=True)

        try:
            inference_model = InferenceModel(video.dataset.inference_model_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid inference model configured for this video's dataset."
            ) from exc

        try:
            classification_url: str = get_env_variable("CLASSIFICATION_SERVICE_URL")
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.get(
                    f"{classification_url}/classification/classify_video_gradcam",
                    params={"video_path": video.path, "inference_model": inference_model.value},
                    timeout=500.0,
                )
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=response.text)
                response.raise_for_status()
                data = response.json()
                overlay_video_path = data["video_path"]

                await self._deduct_credits(db_user, inference_action)

                inference_entry = InferenceHistory(
                    video_id=video.id,
                    user_id=current_user.id,
                    credits_used=inference_action.credits
                )
                inference_classification = InferenceHistoryClassification(
                    ground_truth=video.is_violent,
                    prediction=int(data["predicted_label"]),
                    inference_history=inference_entry
                )
                try:
                    await self.inference_history_repository.add_inference_history_classification(
                        inference_classification
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error while adding an inference history classification entry: {str(e)}"
                    ) from e
                should_cleanup = False
                return InferenceVideoResult(
                    video_path=overlay_video_path,
                    predicted_label=data["predicted_label"],
                    confidence=data["confidence"],
                    predicted_class_probability=data["predicted_class_probability"],
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during classification: {str(e)}"
            ) from e
        finally:
            if should_cleanup and overlay_video_path and os.path.exists(overlay_video_path):
                os.remove(overlay_video_path)

    async def people_tracking(self, video_id: int, current_user: User) -> tuple[str, int]:
        output_video_path: str | None = None
        should_cleanup = True
        db_user = await self._get_user_or_404(current_user.id)
        inference_action = await self.inference_actions_repository.get_inference_action_by_action_id(
            Action.PEOPLE_TRACKING
        )
        await self._ensure_credits(db_user, inference_action)
        video = await self._get_video_or_404(video_id)
        people_tracking_url = get_env_variable("PEOPLE_TRACKING_SERVICE_URL")
        try:
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.get(
                    f"{people_tracking_url}/people_tracking",
                    params={"video_path": video.path},
                    timeout=500.0
                )
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=response.text)
                response.raise_for_status()
                data = response.json()
                output_video_path = data["video_path"]

                await self._deduct_credits(db_user, inference_action)

                inference_entry = InferenceHistory(
                    video_id=video.id,
                    user_id=current_user.id,
                    credits_used=inference_action.credits
                )
                inference_classification = InferenceHistoryPeopleTracking(
                    people_tracked=int(data["people_tracked"]),
                    inference_history=inference_entry
                )
                try:
                    await self.inference_history_repository.add_inference_history_classification(
                        inference_classification
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error while adding an inference history classification entry: {str(e)}"
                    ) from e
                should_cleanup = False
                return output_video_path, data["people_tracked"]
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to build browser-compatible tracking video: {exc}",
            ) from exc
        finally:
            if should_cleanup and output_video_path and os.path.exists(output_video_path):
                os.remove(output_video_path)
