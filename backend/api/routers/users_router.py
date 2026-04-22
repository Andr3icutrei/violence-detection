from typing import List, Optional

from fastapi import APIRouter, Depends, status, Query
from fastapi_mail import ConnectionConfig
from sqlalchemy.ext.asyncio import AsyncSession
import os
from dotenv import load_dotenv
from starlette.status import HTTP_200_OK

from api.routers.auth_router import auth_service
from core.database import get_db
from models import User
from schemas.users_schema import CreateUserDto, UserResponseDto, UserBanRequestDto
from services.users_service import UsersService

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)

users_service = UsersService()

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
async def create_user(user_create_data: CreateUserDto, db: AsyncSession = Depends(get_db)):
    return await users_service.create_user(user_create_data, conf, db)

@router.patch("/verify_account", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def verify_account(token: str, db: AsyncSession = Depends(get_db)):
    return await users_service.verify_account(token, db)

@router.patch("/reset_password", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def reset_password(token: str, new_password: str = Query(..., alias="newPassword"), db: AsyncSession = Depends(get_db)):
    return await users_service.reset_password(token, new_password, db)

@router.get("/request_reset_password", status_code=status.HTTP_200_OK)
async def request_reset_password(email: str, db: AsyncSession = Depends(get_db)):
    await users_service.send_reset_password_email(email, conf, db)
    return {"message": "If an account with the provided email exists, a password reset link has been sent."}

@router.get("/verify_reset_password_token", status_code=status.HTTP_200_OK)
async def verify_reset_password_token(token: str, db: AsyncSession = Depends(get_db)):
    return await users_service.verify_reset_password_token(token, db)

@router.get("/resend_verification_email", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def resend_verification_email(token: str, db: AsyncSession = Depends(get_db)):
    return await users_service.resend_verification_email(token, conf, db)

@router.get("/topbar_information", response_model=UserResponseDto, status_code=HTTP_200_OK)
async def get_topbar_information(db: AsyncSession = Depends(get_db), current_user: User = Depends(auth_service.get_current_user)):
    user: User = await users_service.get_user_by_id(current_user.id, db)

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
    current_admin_user: User = Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    return await users_service.get_all_users(search_term, page, page_size, db)

@router.patch("/update_user_role", status_code=status.HTTP_200_OK)
async def update_user_role(
    user_id: int,
    is_admin: bool,
    current_admin_user: User = Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db)
) -> None:
    await users_service.update_user_role(user_id, is_admin, db)

@router.patch("/ban_user", status_code=status.HTTP_200_OK)
async def ban_user(
    user_id: int,
    user_ban_request_dto: UserBanRequestDto,
    current_admin_user: User = Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db)
) -> None:
    await users_service.ban_user(user_id, user_ban_request_dto.ban_reason, conf, db)
