from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi_mail import ConnectionConfig
import os
from dotenv import load_dotenv
from starlette.status import HTTP_200_OK

from api.dependencies import get_users_service
from models import User
from schemas.users_schema import CreateUserDto, UserResponseDto, UserBanRequestDto, UsersStatsResponseDto
from services.auth_service import get_current_user, get_current_admin_user
from services.users_service import UsersService

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)

load_dotenv()

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

@router.post("/create", response_model = UserResponseDto, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_create_data: CreateUserDto,
    users_service: UsersService = Depends(get_users_service)
):
    return await users_service.create_user(user_create_data, conf)

@router.patch("/verify_account", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def verify_account(
    token: str,
    users_service: UsersService = Depends(get_users_service)
):
    return await users_service.verify_account(token)

@router.patch("/reset_password", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def reset_password(
    token: str,
    new_password: str = Query(..., alias="newPassword"),
    users_service: UsersService = Depends(get_users_service)
):
    return await users_service.reset_password(token, new_password)

@router.get("/request_reset_password", status_code=status.HTTP_200_OK)
async def request_reset_password(
    email: str,
    users_service: UsersService = Depends(get_users_service)
):
    await users_service.send_reset_password_email(email, conf)
    return {"message": "If an account with the provided email exists, a password reset link has been sent."}

@router.get("/verify_reset_password_token", status_code=status.HTTP_200_OK)
async def verify_reset_password_token(
    token: str,
    users_service: UsersService = Depends(get_users_service)
):
    return await users_service.verify_reset_password_token(token)

@router.get("/resend_verification_email", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def resend_verification_email(
    token: str,
    users_service: UsersService = Depends(get_users_service)
):
    return await users_service.resend_verification_email(token, conf)

@router.get("/topbar_information", response_model=UserResponseDto, status_code=HTTP_200_OK)
async def get_topbar_information(
    current_user: User = Depends(get_current_user),
    users_service: UsersService = Depends(get_users_service)
):
    user: User = await users_service.get_user_by_id(current_user.id,)
    return UserResponseDto(
        id=user.id,
        email=user.email,
        credits=user.credits,
        is_admin=user.is_admin
    )

@router.get("/get_all_users", response_model=List[UserResponseDto], status_code=status.HTTP_200_OK)
async def get_all_users(
    search_term: Optional[str] = None,
    page: int = 0,
    page_size: int = 10,
    current_admin_user: User = Depends(get_current_admin_user),
    users_service: UsersService = Depends(get_users_service)
):
    return await users_service.get_all_users(search_term, page, page_size)

@router.patch("/update_user_role", status_code=status.HTTP_200_OK)
async def update_user_role(
    user_id: int,
    is_admin: bool,
    current_admin_user: User = Depends(get_current_admin_user),
    users_service: UsersService = Depends(get_users_service)
) -> None:
    await users_service.update_user_role(user_id, is_admin)

@router.patch("/ban_user/{user_id}", status_code=status.HTTP_200_OK)
async def ban_user(
    user_id: int,
    user_ban_request_dto: UserBanRequestDto,
    current_admin_user: User = Depends(get_current_admin_user),
    users_service: UsersService = Depends(get_users_service)
) -> None:
    await users_service.ban_user(user_id, user_ban_request_dto.ban_reason, conf)

@router.get("/get_users_stats", response_model=UsersStatsResponseDto, status_code=status.HTTP_200_OK)
async def get_users_stats(
    users_service: UsersService = Depends(get_users_service),
    current_admin_user: User = Depends(get_current_admin_user)
) -> UsersStatsResponseDto:
    return await users_service.get_users_stats()
