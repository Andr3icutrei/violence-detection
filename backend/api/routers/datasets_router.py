from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from core.database import get_db
from schemas.datasets_schema import DatasetResponseDto, CreateDatasetRequestDto, PendingDatasetResponseDto
from services.auth_service import AuthService
from services.datasets_service import DatasetsService

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)

datasets_service = DatasetsService()
auth_service: AuthService = AuthService()

@router.get("/get_accepted_datasets", response_model=List[DatasetResponseDto], status_code=status.HTTP_200_OK)
async def get_accepted_datasets(current_user = Depends(auth_service.get_current_user), db: AsyncSession = Depends(get_db)):
    return await datasets_service.get_accepted_datasets(db)

@router.post("/create_unofficial_dataset", status_code=status.HTTP_200_OK)
async def create_unofficial_dataset(
    create_dataset_dto: CreateDatasetRequestDto = Depends(CreateDatasetRequestDto.as_form),
    current_user = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    await datasets_service.create_unofficial_dataset(db, create_dataset_dto, current_user.id)

@router.get("/get_pending_datasets", response_model=List[PendingDatasetResponseDto], status_code=status.HTTP_200_OK)
async def get_pending_datasets(
    search_term: str,
    page: int = 1,
    page_size: int = 10,
    current_user = Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    return await datasets_service.get_pending_datasets(db, search_term, page, page_size)