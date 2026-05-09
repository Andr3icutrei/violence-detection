from fastapi import APIRouter, Depends
from starlette import status

from api.dependencies import get_credits_service
from services.auth_service import get_current_admin_user
from services.credits_service import CreditsService

router=APIRouter(
    prefix="/credits",
    tags=["Credits"],
)

@router.get("/get_credits_cronjob_update", status_code=status.HTTP_200_OK)
async def get_credits_cronjob_update(
    current_admin_user = Depends(get_current_admin_user),
    credits_service: CreditsService = Depends(get_credits_service)
):
    return await credits_service.get_credits_cronjob_update()

@router.patch("/patch_credits_cronjob_update", status_code=status.HTTP_200_OK)
async def patch_credits_cronjob_update(
    new_credits: int,
    current_admin_user = Depends(get_current_admin_user),
    credits_service: CreditsService = Depends(get_credits_service)
):
    await credits_service.patch_credits_cronjob_update(new_credits)
