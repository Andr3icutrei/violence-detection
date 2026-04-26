import os
from typing import List

from fastapi import APIRouter, Depends
from fastapi_mail import ConnectionConfig
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from api.routers.datasets_ws_router import dataset_updates_ws
from core.database import get_db
from models.dataset_status import DatasetStatus
from schemas.datasets_schema import DatasetResponseDto, CreateDatasetRequestDto, DatasetToReviewResponseDto, \
    DatasetWithVideosResponseDto, ReviewDatasetRequestDto, EditDatasetRequestDto, DatasetsStatsResponseDto
from services.auth_service import AuthService
from services.datasets_service import DatasetsService

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)

datasets_service = DatasetsService(notifier=dataset_updates_ws)
auth_service: AuthService = AuthService()

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_USERNAME"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

@router.get("/get_accepted_datasets", response_model=List[DatasetResponseDto], status_code=status.HTTP_200_OK)
async def get_accepted_datasets(current_user = Depends(auth_service.get_current_user), db: AsyncSession = Depends(get_db)):
    return await datasets_service.get_accepted_datasets(db)

@router.post("/create_unofficial_dataset", status_code=status.HTTP_200_OK)
async def create_unofficial_dataset(
    create_dataset_dto: CreateDatasetRequestDto = Depends(CreateDatasetRequestDto.as_form),
    current_user = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    await datasets_service.create_unofficial_dataset(create_dataset_dto, current_user.id, db)

@router.get("/get_datasets", response_model=List[DatasetToReviewResponseDto], status_code=status.HTTP_200_OK)
async def get_datasets(
    search_term: str | None = None,
    page: int = 1,
    page_size: int = 10,
    dataset_status: DatasetStatus | None = None,
    current_user = Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    return await datasets_service.get_datasets(search_term, page, page_size, dataset_status, db)

@router.get("/get_dataset_videos/{dataset_id}", response_model=DatasetWithVideosResponseDto, status_code=status.HTTP_200_OK)
async def get_dataset_videos(
    dataset_id: int,
    current_admin_user = Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    return await datasets_service.get_dataset_videos(dataset_id, db)

@router.patch("/review_dataset/{dataset_id}", response_model=DatasetResponseDto, status_code=status.HTTP_200_OK)
async def review_dataset(
    dataset_id: int,
    request: ReviewDatasetRequestDto,
    current_admin_user = Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    result = await datasets_service.review_dataset(dataset_id, request.is_approved, request.videos, request.review_comment, conf, db)
    return result

@router.delete("/delete_dataset/{dataset_id}", status_code=status.HTTP_200_OK)
async def delete_dataset(
    dataset_id: int,
    current_admin_user = Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    await datasets_service.delete_dataset(dataset_id, db)

@router.patch("/edit_dataset/{dataset_id}", response_model=DatasetResponseDto, status_code=status.HTTP_200_OK)
async def review_dataset(
    dataset_id: int,
    request: EditDatasetRequestDto,
    current_admin_user=Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    result = await datasets_service.edit_dataset(dataset_id, request.videos, db)
    return result

@router.get("/get_datasets_stats", response_model=DatasetsStatsResponseDto, status_code=status.HTTP_200_OK)
async def get_datasets_stats(
    db: AsyncSession = Depends(get_db),
    current_admin_user = Depends(auth_service.get_current_admin_user)
):
    return await datasets_service.get_datasets_stats(db)