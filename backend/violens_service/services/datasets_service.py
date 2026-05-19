import asyncio
import datetime
from typing import List
import os

import httpx
from fastapi import HTTPException
from fastapi_mail import ConnectionConfig
from starlette import status

from helpers.bucket_helper import create_unofficial_dataset_bucket, get_presigned_url, delete_dataset_videos, \
    get_used_storage_gb, upload_inference_model, delete_inference_model_object
from helpers.env_helper import get_env_variable
from models import Dataset, User, InferenceModel
from models.dataset_status import DatasetStatus
from notifier.dataset_events_notifier import DatasetEventsNotifier
from repositories.datasets_repository import DatasetsRepository
from repositories.inference_models_repository import InferenceModelsRepository
from repositories.users_repository import UsersRepository
from repositories.videos_repository import VideosRepository
from schemas.datasets_schema import DatasetResponseDto, CreateDatasetRequestDto, DatasetToReviewResponseDto, \
    DatasetWithVideosResponseDto, DatasetsStatsResponseDto, MostPopularDatasetResponseDto, ValidateModelResponseDto, \
    ConfusionMatrixDto
from schemas.users_schema import UserResponseDto
from schemas.videos_schema import VideoResponseDto, ReviewVideoRequestDto
from helpers.email_helper import send_dataset_approval_mail, send_dataset_rejection_mail


class DatasetsService:
    def __init__(
        self,
        datasets_repository: DatasetsRepository,
        users_repository: UsersRepository,
        videos_repository: VideosRepository,
        inference_models_repository: InferenceModelsRepository,
        notifier: DatasetEventsNotifier
    ):
        self.datasets_repository = datasets_repository
        self.users_repository = users_repository
        self.videos_repository = videos_repository
        self.inference_models_repository = inference_models_repository
        self.notifier = notifier

    async def get_accepted_datasets(self) -> List[DatasetResponseDto]:
        result: List[Dataset] = await self.datasets_repository.get_all_accepted()
        return [
            DatasetResponseDto(
                id=dataset.id,
                name=dataset.name,
                is_official=dataset.is_official,
                status=dataset.status
            ) for dataset in result
        ]

    async def get_datasets(
        self,
        search_term: str | None,
        page: int,
        page_size: int,
        dataset_status: DatasetStatus | None,
        is_official: bool | None
    ) -> List[DatasetToReviewResponseDto]:
        result: List[Dataset] = await self.datasets_repository.get_all(
            search_term=search_term,
            page=page,
            page_size=page_size,
            dataset_status=dataset_status,
            is_official=is_official
        )
        return [
            DatasetToReviewResponseDto(
                id=dataset.id,
                name=dataset.name,
                is_official=dataset.is_official,
                user=UserResponseDto(
                    id=dataset.created_by_user.id,
                    email=dataset.created_by_user.email,
                    is_admin=dataset.created_by_user.is_admin,
                ),
                status=dataset.status,
                videos_count=len(dataset.videos),
                violent_videos_count=len([video for video in dataset.videos if video.is_violent]),
                non_violent_videos_count=len([video for video in dataset.videos if not video.is_violent])
            ) for dataset in result
        ]

    async def create_unofficial_dataset(self, create_dataset_dto: CreateDatasetRequestDto, user_id: int) -> None:
        model_record: InferenceModel | None = None
        already_existing_dataset: Dataset | None = await self.datasets_repository.get_by_name(create_dataset_dto.name)
        if already_existing_dataset is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "DATASET_NAME_EXISTS",
                    "message": f"Dataset with name '{create_dataset_dto.name}' already exists."
                }
            )
        user: User = await self.users_repository.get_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found."
            )
        is_official = bool(user.is_admin)
        try:
            user_has_pending_datasets: bool = await self.datasets_repository.user_has_pending_datasets(user_id)
            if user_has_pending_datasets:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error_code": "USER_HAS_PENDING_DATASETS",
                        "message": "User has pending datasets. Please wait until they are processed before creating a new one."
                    }
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to check pending datasets: {str(e)}"
            )
        await create_unofficial_dataset_bucket(create_dataset_dto.name, create_dataset_dto.videos)
        try:
            if create_dataset_dto.inference_model is None or not create_dataset_dto.inference_model_name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Inference model file and name are required."
                )
            model_path = await upload_inference_model(create_dataset_dto.name, create_dataset_dto.inference_model)
            model_name = create_dataset_dto.inference_model_name
            model_record = await self.inference_models_repository.create(model_name, model_path)
            await self.datasets_repository.create_unofficial_dataset(
                create_dataset_dto.name,
                create_dataset_dto.videos,
                user_id,
                inference_model_id=model_record.id,
                is_official=is_official,
            )
        except Exception as e:
            if model_record is not None:
                try:
                    await delete_inference_model_object(model_record.path)
                    await self.inference_models_repository.delete(model_record)
                except Exception:
                    pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create dataset: {str(e)}"
            )

    async def get_dataset_videos(self, dataset_id: int) -> DatasetWithVideosResponseDto:
        dataset: Dataset | None = await self.datasets_repository.get_by_id_with_videos(dataset_id)
        if dataset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found."
            )
        tasks = [get_presigned_url(video.path) for video in dataset.videos]
        presigned_urls = await asyncio.gather(*tasks)
        for video, url in zip(dataset.videos, presigned_urls):
            video.path = url
        return DatasetWithVideosResponseDto(
            id=dataset.id,
            name=dataset.name,
            is_official=dataset.is_official,
            status=dataset.status,
            videos=[
                VideoResponseDto(
                    id=video.id,
                    uid=str(video.uid),
                    name=video.name,
                    path=video.path,
                    is_violent=video.is_violent,
                    dataset_id=video.dataset_id,
                    dataset_name=video.dataset.name,
                    duration=video.duration,
                    frame_rate=int(video.frame_rate),
                    dataset_is_official=video.dataset.is_official
                ) for video in dataset.videos
            ]
        )

    async def review_dataset(self,
        dataset_id: int,
        is_approved: bool,
        videos: List[ReviewVideoRequestDto],
        review_comment: str,
        conf: ConnectionConfig,
    ) -> Dataset:
        dataset: Dataset | None = await self.datasets_repository.get_by_id_with_videos(dataset_id)
        if dataset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found."
            )
        if dataset.status is not DatasetStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Dataset already reviewed."
            )
        dataset.comment = review_comment
        result = None
        if is_approved:
            result = await self._accept_dataset(dataset, videos)
            try:
                await send_dataset_approval_mail(str(dataset.created_by_user.email), str(dataset.name), review_comment, conf)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to send approval email: {str(e)}"
                )
        else:
            result = await self._reject_dataset(dataset, videos)
            try:
                await delete_dataset_videos(dataset.name)
                await send_dataset_rejection_mail(str(dataset.created_by_user.email), str(dataset.name), review_comment, conf)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to delete dataset videos or send rejection email: {str(e)}"
                )
        await self.notifier.broadcast_dataset_updated(dataset_id=result.id)
        return result

    async def _accept_dataset(self, dataset: Dataset, videos: List[ReviewVideoRequestDto]) -> Dataset:
        for video in dataset.videos:
            review_video_dto = next((v for v in videos if v.video_id == video.id), None)
            if review_video_dto is not None:
                video.is_violent = review_video_dto.is_violent
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video with id {video.id} is missing in the review request."
                )
        dataset.status = DatasetStatus.ACCEPTED
        return await self.datasets_repository.save(dataset)

    async def _reject_dataset(self, dataset: Dataset, videos: List[ReviewVideoRequestDto]) -> Dataset:
        dataset.status = DatasetStatus.REJECTED
        review_video_ids = {v.video_id for v in videos}
        for video in dataset.videos:
            if video.id not in review_video_ids:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video with id {video.id} is missing in the review request."
                )
        for video in list(dataset.videos):
            dataset.videos.remove(video)
            await self.videos_repository.delete(video)
        inference_model_id = dataset.inference_model_id
        await self.datasets_repository.delete(dataset)
        if inference_model_id:
            await self._delete_inference_model_if_unassigned(inference_model_id)
        return dataset

    async def delete_dataset(self, dataset_id: int) -> None:
        dataset: Dataset | None = await self.datasets_repository.get_by_id_with_videos(dataset_id)
        if dataset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found."
            )
        try:
            inference_model_id = dataset.inference_model_id
            await delete_dataset_videos(dataset.name)
            for video in list(dataset.videos):
                dataset.videos.remove(video)
                await self.videos_repository.delete(video)
            await self.datasets_repository.delete(dataset)
            if inference_model_id:
                await self._delete_inference_model_if_unassigned(inference_model_id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete dataset videos or dataset record: {str(e)}"
            )

    async def edit_dataset(self, dataset_id: int, videos: List[ReviewVideoRequestDto]) -> Dataset:
        dataset: Dataset | None = await self.datasets_repository.get_by_id_with_videos(dataset_id)
        if dataset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found."
            )
        if dataset.is_official:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Official datasets cannot be edited."
            )
        for video in dataset.videos:
            review_video_dto = next((v for v in videos if v.video_id == video.id), None)
            if review_video_dto is not None:
                video.is_violent = review_video_dto.is_violent
                await self.videos_repository.save(video)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video with id {video.id} is missing in the edit request."
                )
        await self.notifier.broadcast_dataset_updated(dataset_id=dataset.id)
        return await self.datasets_repository.save(dataset)

    async def validate_dataset_model(self, dataset_id: int, videos: List[ReviewVideoRequestDto]) -> ValidateModelResponseDto:
        dataset: Dataset | None = await self.datasets_repository.get_by_id_with_videos(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
        
        model_record = None
        if dataset.inference_model_id:
            model_record = await self.inference_models_repository.get_by_id(dataset.inference_model_id)

        if not model_record:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inference model not found for this dataset.")

        classification_url: str = get_env_variable("CLASSIFICATION_SERVICE_URL")
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        async with httpx.AsyncClient(verify=False) as client:
            for review_video_dto in videos:
                db_video = next((v for v in dataset.videos if v.id == review_video_dto.video_id), None)
                if not db_video:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Video with id {review_video_dto.video_id} not found in dataset.")
                
                try:
                    response = await client.get(
                        f"{classification_url}/classification/classify_video",
                        params={"video_path": db_video.path, "inference_model_path": model_record.path},
                        timeout=500.0,
                    )
                    if response.status_code != 200:
                        raise HTTPException(status_code=response.status_code, detail=response.text)
                    
                    data = response.json()
                    predicted_label = str(data.get("predicted_label", "")).lower()
                    
                    is_violent_pred = "viol" in predicted_label and "non" not in predicted_label
                    is_violent_gt = review_video_dto.is_violent

                    if is_violent_gt and is_violent_pred:
                        tp += 1
                    elif not is_violent_gt and not is_violent_pred:
                        tn += 1
                    elif not is_violent_gt and is_violent_pred:
                        fp += 1
                    elif is_violent_gt and not is_violent_pred:
                        fn += 1
                except Exception as e:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Classification failed for video {db_video.id}: {str(e)}")
                    
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return ValidateModelResponseDto(
            accuracy=accuracy,
            confusion_matrix=ConfusionMatrixDto(
                true_positive=tp,
                true_negative=tn,
                false_positive=fp,
                false_negative=fn
            )
        )

    async def _delete_inference_model_if_unassigned(self, inference_model_id: int) -> None:
        if not inference_model_id:
            return
        count = await self.inference_models_repository.count_datasets(inference_model_id)
        if count > 0:
            return
        model = await self.inference_models_repository.get_by_id(inference_model_id)
        if model is None:
            return
        await delete_inference_model_object(model.path)
        await self.inference_models_repository.delete(model)

    async def get_datasets_stats(self) -> DatasetsStatsResponseDto:
        most_popular_dataset_classification, classification_videos_count = await self.datasets_repository.get_most_popular_dataset_classification() or (None, 0)
        most_popular_dataset_people_tracking, people_tracking_videos_count = await self.datasets_repository.get_most_popular_dataset_people_tracking() or (None, 0)
        official_datasets_count: int = await self.datasets_repository.get_official_datasets_count()
        unofficial_datasets_count: int = await self.datasets_repository.get_unofficial_datasets_count()
        pending_datasets_count: int = await self.datasets_repository.get_pending_datasets_count()
        storage_used_gb = await get_used_storage_gb()

        return DatasetsStatsResponseDto(
            most_popular_dataset_classification=MostPopularDatasetResponseDto(
                id=most_popular_dataset_classification.id,
                name=most_popular_dataset_classification.name,
                is_official=most_popular_dataset_classification.is_official,
                status=most_popular_dataset_classification.status,
                inferences_videos_count=classification_videos_count
            ) if most_popular_dataset_classification else None,
            most_popular_dataset_people_tracking=MostPopularDatasetResponseDto(
                id=most_popular_dataset_people_tracking.id,
                name=most_popular_dataset_people_tracking.name,
                is_official=most_popular_dataset_people_tracking.is_official,
                status=most_popular_dataset_people_tracking.status,
                inferences_videos_count=people_tracking_videos_count
            ) if most_popular_dataset_people_tracking else None,
            official_datasets_count=official_datasets_count,
            unofficial_datasets_count=unofficial_datasets_count,
            pending_datasets_count=pending_datasets_count,
            storage_used_gb=storage_used_gb
        )



