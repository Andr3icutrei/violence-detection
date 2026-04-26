import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from core.database import get_db
from schemas.inference_history_schema import InferenceHistoryStatsResponseDto
from services.auth_service import AuthService
from services.inference_history_service import InferenceHistoryService

router = APIRouter(
    prefix="/inference_history",
    tags=["Inference history"],
)

auth_service = AuthService()
inference_history_service = InferenceHistoryService()

@router.get("/get_inference_history_stats")
async def get_inference_history_stats(
    year: Optional[int] = None,
    month: Optional[int] = None,
    admin_user = Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> InferenceHistoryStatsResponseDto:
    now = datetime.datetime.now()
    year = year if year is not None else now.year
    month = month if month is not None else now.month
    return await inference_history_service.get_inference_history_stats(year, month, db)