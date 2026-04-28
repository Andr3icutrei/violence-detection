from fastapi import APIRouter, Depends
from starlette import status

from services.auth_service import AuthService
from services.credits_service import CreditsService

router=APIRouter(
    prefix="/credits",
    tags=["Credits"],
)

auth_service = AuthService()
credits_service = CreditsService()

@router.get("/get_credits_cronjob_update", status_code=status.HTTP_200_OK)
async def get_credits_cronjob_update(
    current_admin_user = Depends(auth_service.get_current_admin_user)
):
    return await credits_service.get_credits_cronjob_update()

@router.patch("/patch_credits_cronjob_update", status_code=status.HTTP_200_OK)
async def patch_credits_cronjob_update(
    new_credits: int,
    current_admin_user = Depends(auth_service.get_current_admin_user)
):
    await credits_service.patch_credits_cronjob_update(new_credits)