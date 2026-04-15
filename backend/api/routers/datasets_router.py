from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from core.database import get_db
from schemas.datasets_schema import DatasetResponseDto, CreateDatasetRequestDto
from services.auth_service import AuthService
from services.datasets_service import DatasetsService

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)

datasets_service = DatasetsService()
auth_service: AuthService = AuthService()

@router.get("/get_datasets", response_model=List[DatasetResponseDto], status_code=status.HTTP_200_OK)
async def get_datasets(current_user = Depends(auth_service.get_current_user), db: AsyncSession = Depends(get_db)):
    return await datasets_service.get_datasets(db)

@router.post("/create_unofficial_dataset", status_code=status.HTTP_200_OK)
async def create_unofficial_dataset(
    create_dataset_dto: CreateDatasetRequestDto = Depends(CreateDatasetRequestDto.as_form),
    current_user = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    await datasets_service.create_unofficial_dataset(db, create_dataset_dto, current_user.id)

