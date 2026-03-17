from fastapi import APIRouter, Depends, status
from fastapi.openapi.utils import status_code_ranges
from fastapi_mail import ConnectionConfig
from sqlalchemy.ext.asyncio import AsyncSession
import os
from dotenv import load_dotenv

from core.database import get_db
from schemas.users_schema import CreateUserDto, UserResponseDto
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
    return await users_service.create_user(db, user_create_data, conf)

@router.get("/{user_id}", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    return await users_service.get_user_by_id(db, user_id)

@router.patch("/verify-account/z", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def verify_account(token: str, db: AsyncSession = Depends(get_db)):
    return await users_service.verify_account(db, token)
