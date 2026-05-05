import asyncio
from typing import List

from fastapi import HTTPException
from fastapi_mail import ConnectionConfig
from starlette import status

from helpers.bucket_helper import create_unofficial_dataset_bucket, get_presigned_url, delete_dataset_videos, \
    get_used_storage_gb
from helpers.email_helper import send_dataset_approval_mail, send_dataset_rejection_mail
from models import Dataset, User
from models.dataset_status import DatasetStatus
from notifier.dataset_events_notifier import DatasetEventsNotifier
from repositories.datasets_repository import DatasetsRepository
from repositories.users_repository import UsersRepository
from repositories.videos_repository import VideosRepository
from schemas.datasets_schema import DatasetResponseDto, CreateDatasetRequestDto, DatasetToReviewResponseDto, \
    DatasetWithVideosResponseDto, DatasetsStatsResponseDto, MostPopularDatasetResponseDto
from schemas.users_schema import UserResponseDto
from schemas.videos_schema import VideoResponseDto, ReviewVideoRequestDto


class DatasetsService:
    def __init__(
        self,
        datasets_repository: DatasetsRepository,
        users_repository: UsersRepository,
        videos_repository: VideosRepository,
        notifier: DatasetEventsNotifier
    ):
        self.datasets_repository = datasets_repository
        self.users_repository = users_repository
        self.videos_repository = videos_repository
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
    ) -> List[DatasetToReviewResponseDto]:
        result: List[Dataset] = await self.datasets_repository.get_all(
            search_term=search_term,
            page=page,
            page_size=page_size,
            dataset_status=dataset_status,
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
                videos_count=len(dataset.videos)
            ) for dataset in result
        ]

    async def create_unofficial_dataset(self, create_dataset_dto: CreateDatasetRequestDto, user_id: int) -> None:
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
            await self.datasets_repository.create_unofficial_dataset(
                create_dataset_dto.name,
                create_dataset_dto.videos,
                user_id,
            )
        except Exception as e:
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
        if dataset.is_official:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Official datasets cannot be reviewed."
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
        if dataset.is_official:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only official datasets can be reviewed."
            )
        dataset.comment = review_comment
        result = None
        if is_approved:
            result = await self._accept_dataset(dataset, videos)
            try:
                await send_dataset_approval_mail(dataset.created_by_user.email, dataset.name, review_comment, conf)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to send approval email: {str(e)}"
                )
        else:
            result = await self._reject_dataset(dataset, videos)
            try:
                await delete_dataset_videos(dataset.name)
                await send_dataset_rejection_mail(dataset.created_by_user.email, dataset.name, review_comment, conf)
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
        for video in dataset.videos:
            review_video_dto = next((v for v in videos if v.video_id == video.id), None)
            if review_video_dto is not None:
                await self.videos_repository.delete(video)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video with id {video.id} is missing in the review request."
                )
        return await self.datasets_repository.save(dataset)

    async def delete_dataset(self, dataset_id: int) -> None:
        dataset: Dataset | None = await self.datasets_repository.get_by_id_with_videos(dataset_id)
        if dataset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found."
            )
        if dataset.is_official:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Official datasets cannot be deleted."
            )
        try:
            await delete_dataset_videos(dataset.name)
            for video in dataset.videos:
                await self.videos_repository.delete(video)
            await self.datasets_repository.delete(dataset)
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