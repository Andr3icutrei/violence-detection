import asyncio
from datetime import datetime
from typing import List

from fastapi import HTTPException
from fastapi_mail import ConnectionConfig
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from helpers.bucket_helper import create_unofficial_dataset_bucket, get_presigned_url, delete_dataset_videos
from helpers.email_helper import send_dataset_approval_mail, send_dataset_rejection_mail
from models import Dataset, User, user
from models.dataset_status import DatasetStatus
from repositories.datasets_repository import DatasetsRepository
from repositories.users_repository import UsersRepository
from schemas.datasets_schema import DatasetResponseDto, CreateDatasetRequestDto, DatasetToReviewResponseDto, \
    DatasetWithVideosResponseDto
from schemas.users_schema import UserResponseDto
from schemas.videos_schema import VideoResponseDto, ReviewVideoRequestDto


class DatasetsService:
    def __init__(self):
        self.datasets_repository = DatasetsRepository()
        self.users_repository = UsersRepository()

    async def get_accepted_datasets(self, db: AsyncSession) -> List[DatasetResponseDto]:
        result: List[Dataset] = await self.datasets_repository.get_all_accepted(db)
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
        db: AsyncSession,
        search_term: str,
        page: int,
        page_size: int,
        dataset_status: DatasetStatus | None = None
    ) -> List[DatasetToReviewResponseDto]:
        result: List[Dataset] = await self.datasets_repository.get_all(db, search_term=search_term, page=page, page_size=page_size, dataset_status=dataset_status)
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

    async def create_unofficial_dataset(self, db: AsyncSession, create_dataset_dto: CreateDatasetRequestDto, user_id: int) -> None:
        already_existing_dataset: Dataset | None = await self.datasets_repository.get_by_name(db, create_dataset_dto.name)
        if already_existing_dataset is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "DATASET_NAME_EXISTS",
                    "message": f"Dataset with name '{create_dataset_dto.name}' already exists."
                }
            )
        user: User = await self.users_repository.get_by_id(db, user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found."
            )
        try:
            user_has_pending_datasets: bool = await self.datasets_repository.user_has_pending_datasets(db, user_id)
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
                db,
                create_dataset_dto.name,
                create_dataset_dto.videos,
                user_id
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create dataset: {str(e)}"
            )

    async def get_dataset_videos(self, db: AsyncSession, dataset_id: int) -> DatasetWithVideosResponseDto:
        dataset: Dataset | None = await self.datasets_repository.get_by_id_with_videos(db, dataset_id)
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
        db: AsyncSession,
        dataset_id: int,
        is_approved: bool,
        videos: List[ReviewVideoRequestDto],
        review_comment: str,
        conf: ConnectionConfig
    ) -> Dataset:
        dataset: Dataset | None = await self.datasets_repository.get_by_id_with_videos(db, dataset_id)
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
            result = await self._accept_dataset(db, dataset, videos)
            try:
                await send_dataset_approval_mail(dataset.created_by_user.email, dataset.name, review_comment, conf)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to send approval email: {str(e)}"
                )
        else:
            result = await self._reject_dataset(db, dataset, videos)
            try:
                await delete_dataset_videos(dataset.name)
                await send_dataset_rejection_mail(dataset.created_by_user.email, dataset.name, review_comment, conf)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to delete dataset videos or send rejection email: {str(e)}"
                )
        return result

    async def _accept_dataset(self, db: AsyncSession, dataset: Dataset, videos: List[ReviewVideoRequestDto]) -> Dataset:
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
        return await self.datasets_repository.save(db, dataset)

    async def _reject_dataset(self, db: AsyncSession, dataset: Dataset, videos: List[ReviewVideoRequestDto]) -> Dataset:
        dataset.status = DatasetStatus.REJECTED
        dataset.deleted_at = datetime.now()
        for video in dataset.videos:
            review_video_dto = next((v for v in videos if v.video_id == video.id), None)
            if review_video_dto is not None:
                video.deleted_at = datetime.now()
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video with id {video.id} is missing in the review request."
                )
        return await self.datasets_repository.save(db, dataset)

