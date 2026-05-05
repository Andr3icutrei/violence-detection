import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from starlette import status

from api.dependencies import get_inference_history_service
from schemas.inference_history_schema import InferenceHistoryStatsResponseDto
from services.auth_service import get_current_admin_user
from services.inference_history_service import InferenceHistoryService

router = APIRouter(
    prefix="/inference_history",
    tags=["Inference history"],
)

@router.get("/get_inference_history_stats", response_model=InferenceHistoryStatsResponseDto, status_code=status.HTTP_200_OK)
async def get_inference_history_stats(
    year: Optional[int] = None,
    month: Optional[int] = None,
    admin_user = Depends(get_current_admin_user),
    inference_history_service: InferenceHistoryService = Depends(get_inference_history_service),
) -> InferenceHistoryStatsResponseDto:
    now = datetime.datetime.now()
    year = year if year is not None else now.year
    month = month if month is not None else now.month
    return await inference_history_service.get_inference_history_stats(year, month)
