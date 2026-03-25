from fastapi import APIRouter, status, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from google.oauth2 import id_token
from google.auth.transport import requests

from core.database import get_db
from helpers.jwt_helper import create_jwt_token
from models.user import User
from schemas.auth_schema import LoginRequestDto, LogoutResponseDto, TokenSchema
from schemas.users_schema import UserResponseDto
from services.auth_service import AuthService

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

authService: AuthService = AuthService()

ACCESS_TOKEN_COOKIE = "access_token"
ACCESS_TOKEN_MAX_AGE_SECONDS = 3600

@router.post("/login", response_model=UserResponseDto, status_code=status.HTTP_201_CREATED)
async def login(
    login_dto: LoginRequestDto,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    user: User = await authService.login(db, login_dto.email, login_dto.password)
    token_payload = {
        "sub": str(user.id),
        "email": user.email,
        "is_admin": user.is_admin,
    }
    jwt_token: str = create_jwt_token(token_payload, "SECRET_JWT_KEY", expires=60)
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE,
        value=jwt_token,
        httponly=True,
        max_age=ACCESS_TOKEN_MAX_AGE_SECONDS,
        samesite="lax",
        secure=True
    )
    return user

@router.post("/google-login", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def login_google(data: TokenSchema, response: Response, db: AsyncSession = Depends(get_db)):
    user: User = await authService.login_google(data, db)
    token_payload = {
        "sub": str(user.id),
        "email": user.email,
        "is_admin": user.is_admin,
    }
    jwt_token: str = create_jwt_token(token_payload, "SECRET_JWT_KEY", expires=60)
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE,
        value=jwt_token,
        httponly=True,
        max_age=ACCESS_TOKEN_MAX_AGE_SECONDS,
        samesite="lax",
        secure=True
    )
    return user

@router.post("/logout", response_model=LogoutResponseDto, status_code=status.HTTP_200_OK)
async def logout(
    response: Response,
    user: User = Depends(authService.get_current_user)
):
    response.delete_cookie(
        key=ACCESS_TOKEN_COOKIE,
        samesite="lax",
        secure=True,
    )
    return { "message": "Logout successful." }

@router.get("/me", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def get_current_logged_user(
    user: User = Depends(authService.get_current_user)
):
    return user