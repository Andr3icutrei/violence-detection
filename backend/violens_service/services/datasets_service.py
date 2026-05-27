import asyncio
from typing import List

from fastapi import HTTPException
from fastapi_mail import ConnectionConfig
from starlette import status

from helpers.bucket_helper import create_unofficial_dataset_bucket, get_presigned_url, delete_dataset_videos, \
    get_used_storage_gb, upload_inference_model, delete_inference_model_object, delete_dataset_video_objects
from helpers.classification_label_helper import is_violent_label
from helpers.env_helper import get_env_variable, get_env_float
from helpers.email_helper import send_dataset_approval_mail, send_dataset_rejection_mail
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
from services.classification_client import ClassificationClient
from services.dataset_validation import build_review_video_map, validate_excluded_ids, validate_review_ids, \
    get_review_video_or_raise, ensure_all_videos_reviewed
from services.model_validation import ConfusionMatrixCounts


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

    def _dataset_to_review_dto(self, dataset: Dataset) -> DatasetToReviewResponseDto:
        return DatasetToReviewResponseDto(
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
        )

    def _video_to_response_dto(self, video) -> VideoResponseDto:
        return VideoResponseDto(
            id=video.id,
            uid=str(video.uid),
            name=video.name,
            path=video.path,
            is_violent=video.is_violent,
            dataset_id=video.dataset_id,
            dataset_name=video.dataset.name,
            duration=video.duration,
            frame_rate=int(video.frame_rate),
            dataset_is_official=video.dataset.is_official,
        )

    async def _get_dataset_or_404(self, dataset_id: int) -> Dataset:
        dataset: Dataset | None = await self.datasets_repository.get_by_id_with_videos(dataset_id)
        if dataset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found."
            )
        return dataset

    async def _set_videos_presigned_urls(self, dataset: Dataset) -> None:
        tasks = [get_presigned_url(video.path) for video in dataset.videos]
        presigned_urls = await asyncio.gather(*tasks)
        for video, url in zip(dataset.videos, presigned_urls):
            video.path = url

    def _get_inference_model_info(self, dataset: Dataset) -> tuple[str | None, str | None]:
        if not dataset.inference_model:
            return None, None
        return dataset.inference_model.name, dataset.inference_model.path

    def _ensure_dataset_pending(self, dataset: Dataset) -> None:
        if dataset.status is not DatasetStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Dataset already reviewed."
            )

    async def _send_approval_email(self, dataset: Dataset, review_comment: str, conf: ConnectionConfig) -> None:
        try:
            await send_dataset_approval_mail(str(dataset.created_by_user.email), str(dataset.name), review_comment, conf)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to send approval email: {str(e)}"
            )

    async def _send_rejection_cleanup(self, dataset: Dataset, review_comment: str, conf: ConnectionConfig) -> None:
        try:
            await delete_dataset_videos(dataset.name)
            await send_dataset_rejection_mail(str(dataset.created_by_user.email), str(dataset.name), review_comment, conf)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete dataset videos or send rejection email: {str(e)}"
            )

    def _ensure_inference_model_payload(self, create_dataset_dto: CreateDatasetRequestDto) -> None:
        if create_dataset_dto.inference_model is None or not create_dataset_dto.inference_model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inference model file and name are required."
            )

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
        return [self._dataset_to_review_dto(dataset) for dataset in result]

    async def create_unofficial_dataset(self, create_dataset_dto: CreateDatasetRequestDto, user_id: int) -> None:
        await self._ensure_dataset_name_available(create_dataset_dto.name)
        await self._get_user_or_404(user_id)
        await self._ensure_user_has_no_pending_datasets(user_id)
        await create_unofficial_dataset_bucket(create_dataset_dto.name, create_dataset_dto.videos)
        await self._create_unofficial_dataset_record(create_dataset_dto, user_id, is_official=False)

    async def _ensure_dataset_name_available(self, dataset_name: str) -> None:
        already_existing_dataset: Dataset | None = await self.datasets_repository.get_by_name(dataset_name)
        if already_existing_dataset is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "DATASET_NAME_EXISTS",
                    "message": f"Dataset with name '{dataset_name}' already exists."
                }
            )

    async def _get_user_or_404(self, user_id: int) -> User:
        user: User = await self.users_repository.get_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found."
            )
        return user

    async def _ensure_user_has_no_pending_datasets(self, user_id: int) -> None:
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

    async def _create_unofficial_dataset_record(
        self,
        create_dataset_dto: CreateDatasetRequestDto,
        user_id: int,
        is_official: bool,
    ) -> None:
        model_record: InferenceModel | None = None
        try:
            self._ensure_inference_model_payload(create_dataset_dto)
            model_record = await self._create_inference_model_record(create_dataset_dto)
            await self.datasets_repository.create_unofficial_dataset(
                create_dataset_dto.name,
                create_dataset_dto.videos,
                user_id,
                inference_model_id=model_record.id,
                is_official=is_official,
            )
        except Exception as e:
            await self._cleanup_inference_model_record(model_record)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create dataset: {str(e)}"
            )

    async def _cleanup_inference_model_record(self, model_record: InferenceModel | None) -> None:
        if model_record is None:
            return
        try:
            await delete_inference_model_object(model_record.path)
            await self.inference_models_repository.delete(model_record)
        except Exception:
            pass

    async def _create_inference_model_record(self, create_dataset_dto: CreateDatasetRequestDto) -> InferenceModel:
        model_path = await upload_inference_model(create_dataset_dto.name, create_dataset_dto.inference_model)
        model_name = create_dataset_dto.inference_model_name
        return await self.inference_models_repository.create(model_name, model_path)

    async def _get_inference_model_or_400(self, dataset: Dataset) -> InferenceModel:
        model_record = None
        if dataset.inference_model_id:
            model_record = await self.inference_models_repository.get_by_id(dataset.inference_model_id)
        if not model_record:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inference model not found for this dataset.")
        return model_record

    def _build_validation_inputs(
        self,
        dataset: Dataset,
        videos: List[ReviewVideoRequestDto],
        excluded_video_ids: List[int] | None,
    ) -> tuple[list, dict, list]:
        excluded_ids = set(excluded_video_ids or [])
        dataset_video_map = {v.id: v for v in dataset.videos}
        validate_excluded_ids(set(dataset_video_map.keys()), excluded_ids)
        review_video_map = build_review_video_map(videos)
        validate_review_ids(set(dataset_video_map.keys()), set(review_video_map.keys()))
        videos_to_validate = [v for v in dataset.videos if v.id not in excluded_ids]
        if not videos_to_validate:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No videos left to validate after exclusions."
            )
        return videos_to_validate, review_video_map, excluded_ids

    async def _run_model_validation(
        self,
        model_record: InferenceModel,
        videos_to_validate: list,
        review_video_map: dict,
    ) -> ConfusionMatrixCounts:
        classification_url: str = get_env_variable("CLASSIFICATION_SERVICE_URL")
        timeout_seconds = get_env_float("CLASSIFICATION_REQUEST_TIMEOUT_SECONDS", 500.0)
        counts = ConfusionMatrixCounts()

        async with ClassificationClient(classification_url, timeout_seconds) as client:
            for db_video in videos_to_validate:
                review_video_dto = get_review_video_or_raise(review_video_map, db_video.id, "validation")
                try:
                    data = await client.classify_video(db_video.path, model_record.path)
                    predicted_label = str(data.get("predicted_label", ""))
                    is_violent_pred = is_violent_label(predicted_label)
                    counts.update(review_video_dto.is_violent, is_violent_pred)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Classification failed for video {db_video.id}: {str(e)}"
                    )

        return counts

    def _build_validate_model_response(self, counts: ConfusionMatrixCounts) -> ValidateModelResponseDto:
        return ValidateModelResponseDto(
            accuracy=counts.accuracy(),
            confusion_matrix=ConfusionMatrixDto(
                true_positive=counts.true_positive,
                true_negative=counts.true_negative,
                false_positive=counts.false_positive,
                false_negative=counts.false_negative
            )
        )

    async def get_dataset_videos(self, dataset_id: int) -> DatasetWithVideosResponseDto:
        dataset = await self._get_dataset_or_404(dataset_id)
        await self._set_videos_presigned_urls(dataset)
        inference_model_name, inference_model_path = self._get_inference_model_info(dataset)
        return DatasetWithVideosResponseDto(
            id=dataset.id,
            name=dataset.name,
            is_official=dataset.is_official,
            status=dataset.status,
            inference_model_name=inference_model_name,
            inference_model_path=inference_model_path,
            videos=[self._video_to_response_dto(video) for video in dataset.videos]
        )

    async def review_dataset(self,
        dataset_id: int,
        is_approved: bool,
        videos: List[ReviewVideoRequestDto],
        review_comment: str,
        conf: ConnectionConfig,
        excluded_video_ids: List[int] | None = None,
    ) -> Dataset:
        dataset = await self._get_dataset_or_404(dataset_id)
        self._ensure_dataset_pending(dataset)
        dataset.comment = review_comment
        result = None
        if is_approved:
            result = await self._accept_dataset(dataset, videos, excluded_video_ids)
            await self._send_approval_email(dataset, review_comment, conf)
        else:
            result = await self._reject_dataset(dataset, videos)
            await self._send_rejection_cleanup(dataset, review_comment, conf)
        await self.notifier.broadcast_dataset_updated(dataset_id=result.id)
        return result

    async def _accept_dataset(self, dataset: Dataset, videos: List[ReviewVideoRequestDto], excluded_video_ids: List[int] | None = None) -> Dataset:
        review_video_map = build_review_video_map(videos)
        excluded_ids = set(excluded_video_ids or [])
        dataset_video_ids = {v.id for v in dataset.videos}
        validate_excluded_ids(dataset_video_ids, excluded_ids)
        for video in dataset.videos:
            if video.id in excluded_ids:
                continue
            review_video_dto = get_review_video_or_raise(review_video_map, video.id, "review")
            video.is_violent = review_video_dto.is_violent
        if excluded_ids:
            await self._exclude_videos(dataset, list(excluded_ids))
        dataset.status = DatasetStatus.ACCEPTED
        return await self.datasets_repository.save(dataset)

    async def _reject_dataset(self, dataset: Dataset, videos: List[ReviewVideoRequestDto]) -> Dataset:
        dataset.status = DatasetStatus.REJECTED
        review_video_ids = {v.video_id for v in videos}
        ensure_all_videos_reviewed(dataset.videos, review_video_ids, "review")
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

    async def edit_dataset(self, dataset_id: int, videos: List[ReviewVideoRequestDto], excluded_video_ids: List[int] | None = None) -> Dataset:
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
        review_video_map = build_review_video_map(videos)
        excluded_ids = set(excluded_video_ids or [])
        dataset_video_ids = {v.id for v in dataset.videos}
        validate_excluded_ids(dataset_video_ids, excluded_ids)
        validate_review_ids(dataset_video_ids, set(review_video_map.keys()))
        for video in dataset.videos:
            if video.id in excluded_ids:
                continue
            review_video_dto = get_review_video_or_raise(review_video_map, video.id, "edit")
            video.is_violent = review_video_dto.is_violent
            await self.videos_repository.save(video)
        if excluded_ids:
            await self._exclude_videos(dataset, list(excluded_ids))
        await self.notifier.broadcast_dataset_updated(dataset_id=dataset.id)
        return await self.datasets_repository.save(dataset)

    async def validate_dataset_model(self, dataset_id: int, videos: List[ReviewVideoRequestDto], excluded_video_ids: List[int] | None = None) -> ValidateModelResponseDto:
        dataset = await self._get_dataset_or_404(dataset_id)
        model_record = await self._get_inference_model_or_400(dataset)
        videos_to_validate, review_video_map, _ = self._build_validation_inputs(dataset, videos, excluded_video_ids)
        counts = await self._run_model_validation(model_record, videos_to_validate, review_video_map)
        return self._build_validate_model_response(counts)

    async def _exclude_videos(self, dataset: Dataset, excluded_video_ids: List[int]) -> None:
        if not excluded_video_ids:
            return
        dataset_video_map = {v.id: v for v in dataset.videos}
        object_keys = [dataset_video_map[video_id].path for video_id in excluded_video_ids]
        await delete_dataset_video_objects(object_keys)
        for video_id in excluded_video_ids:
            video = dataset_video_map[video_id]
            dataset.videos.remove(video)
            await self.videos_repository.delete(video)

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


