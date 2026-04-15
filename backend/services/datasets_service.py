from typing import List

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from helpers.bucket_helper import create_unofficial_dataset_bucket
from models import Dataset, User, user
from repositories.datasets_repository import DatasetsRepository
from repositories.users_repository import UsersRepository
from schemas.datasets_schema import DatasetResponseDto, CreateDatasetRequestDto, PendingDatasetResponseDto
from schemas.users_schema import UserResponseDto


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
                is_official=dataset.is_official
            ) for dataset in result
        ]

    async def get_pending_datasets(
        self,
        db: AsyncSession,
        search_term: str,
        page: int,
        page_size: int
    ) -> List[PendingDatasetResponseDto]:
        result: List[Dataset] = await self.datasets_repository.get_all_pending(db, search_term=search_term, page=page, page_size=page_size)
        return [
            PendingDatasetResponseDto(
                id=dataset.id,
                name=dataset.name,
                is_official=dataset.is_official,
                user=UserResponseDto(
                    id=dataset.created_by_user.id,
                    email=dataset.created_by_user.email,
                    is_admin=dataset.created_by_user.is_admin,
                ),
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