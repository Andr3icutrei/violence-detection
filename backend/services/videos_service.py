import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import List, Sequence

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from helpers.bucket_helper import download_object_to_file, get_presigned_url
from helpers.inference_helper import run_people_tracking, run_classification_and_gradcam

from models import Video, InferenceHistory, User
from models.action import Action
from models.inference_action import InferenceAction
from models.inference_history_classification import InferenceHistoryClassification
from models.inference_history_people_tracking import InferenceHistoryPeopleTracking
from models.inference_model import InferenceModel
from repositories.inference_actions_repository import InferenceActionsRepository

from repositories.inference_history_repository import InferenceHistoryRepository
from repositories.users_repository import UsersRepository
from repositories.videos_repository import VideosRepository

from services.inference_runtime import InferenceRuntime

@dataclass
class InferenceVideoResult:
    video_path: str
    predicted_label: str
    confidence: float
    predicted_class_probability: float

class VideosService:
    def __init__(self, inference_runtime: InferenceRuntime):
        self.videos_repository = VideosRepository()
        self.inference_history_repository = InferenceHistoryRepository()
        self.users_repository = UsersRepository()
        self.inference_actions_repository = InferenceActionsRepository()

        self.inference_runtime = inference_runtime

    async def get_videos_paged(
        self,
        search_term: str | None,
        dataset_id: int | None,
        is_violent: bool | None = None,
        asc: bool = True,
        page: int = 0,
        page_size: int = 40,
        *,
        db: AsyncSession,
    ) -> List[Video]:
        videos: Sequence[Video] = await (
            self.videos_repository.get_videos_paged(search_term, dataset_id, is_violent, asc, page, page_size, db=db))

        tasks = [get_presigned_url(video.path) for video in videos]
        presigned_urls = await asyncio.gather(*tasks)

        result = []
        for video, url in zip(videos, presigned_urls):
            video.path = url
            result.append(video)

        return result

    async def exists_video(self, video_uid: str, db: AsyncSession) -> bool:
        video: Video = await self.videos_repository.get_by_uid(video_uid, db)
        return video is not None

    @staticmethod
    def _to_label(is_violent: bool) -> str:
        return "violent" if is_violent else "non-violent"

    async def classify_and_gradcam_video(self, video_id: int, current_user: User, db: AsyncSession) -> InferenceVideoResult:
        db_user = await self.users_repository.get_by_id(current_user.id, db)
        if db_user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found."
            )

        inference_action: InferenceAction = await self.inference_actions_repository.get_inference_action_by_action_id(Action.CLASSIFICATION, db)

        if db_user.credits - inference_action.credits < 0:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Not enough credits to perform this action."
            )

        video: Video = await self.videos_repository.get_by_id_for_classification(video_id, db)
        if video is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found."
            )

        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_path = temp_video_file.name
        temp_video_file.close()

        was_downloaded = await download_object_to_file(video.path, temp_video_path)
        if not was_downloaded:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found in storage."
            )

        inference_model = InferenceModel.INVALID
        try:
            inference_model = InferenceModel(video.dataset.inference_model_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid inference model configured for this video's dataset."
            )

        try:
            overlay_video_path, predicted_label, confidence, predicted_class_probability = run_classification_and_gradcam(
                inference_model,
                temp_video_path,
                self.inference_runtime
            )

            db_user.credits = db_user.credits - inference_action.credits
            try:
                db_user = await self.users_repository.add_user(db_user, db)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error while updating user credits: {str(e)}"
                ) from e

            inference_entry = InferenceHistory(
                video_id=video.id,
                user_id=current_user.id,
                credits_used=inference_action.credits
            )

            inference_classification = InferenceHistoryClassification(
                ground_truth=video.is_violent,
                prediction=predicted_label,
                inference_history=inference_entry
            )
            try:
                await self.inference_history_repository.add_inference_history_classification(inference_classification, db)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error while adding an inference history classification entry: {str(e)}"
                ) from e

            return InferenceVideoResult(
                video_path=overlay_video_path,
                predicted_label=predicted_label,
                confidence=confidence,
                predicted_class_probability=predicted_class_probability,
            )
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

    async def people_tracking(self, video_id: int, current_user: User, db: AsyncSession) -> tuple[str, int]:
        db_user = await self.users_repository.get_by_id(current_user.id, db)
        if db_user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found."
            )

        inference_action: InferenceAction = await self.inference_actions_repository.get_inference_action_by_action_id(Action.PEOPLE_TRACKING, db)

        if db_user.credits -inference_action.credits < 0:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Not enough credits to perform this action."
            )

        video: Video = await self.videos_repository.get_by_id(video_id, db)
        if video is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found."
            )

        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_path = temp_video_file.name
        temp_video_file.close()

        was_downloaded = await download_object_to_file(video.path, temp_video_path)
        if not was_downloaded:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found in storage."
            )

        try:
            video_path, people_count = run_people_tracking(temp_video_path, self.inference_runtime.yolo_model)
            inference_entry = InferenceHistory(
                video_id=video.id,
                user_id=current_user.id,
                credits_used=inference_action.credits
            )

            inference_people_tracking = InferenceHistoryPeopleTracking(
                people_tracked=people_count,
                inference_history=inference_entry
            )
            db_user.credits = db_user.credits - inference_action.credits
            try:
                db_user = await self.users_repository.add_user(db_user, db)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error while updating user credits: {str(e)}"
                ) from e
            await self.inference_history_repository.add_inference_people_tracking(inference_people_tracking, db)

            return video_path, people_count
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to build browser-compatible tracking video: {exc}",
            )
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
