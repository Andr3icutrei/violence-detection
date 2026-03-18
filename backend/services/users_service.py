from jwt import PyJWTError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status
from fastapi_mail import ConnectionConfig

from schemas.users_schema import CreateUserDto, UserResponseDto
from core.security import get_password_hash
from repositories.users_repository import UsersRepository
from models.user import User
from helpers.email_helper import send_registration_email
from helpers.jwt_helper import create_jwt_token, decode_jwt_token, decode_jwt_token_without_exp_check

class UsersService:
    def __init__(self):
        self.users_repository = UsersRepository()

    async def create_user(self, db: AsyncSession, user_create_data: CreateUserDto, conf: ConnectionConfig) -> User:
        db_user: User | None = await self.users_repository.get_by_email(db, email=user_create_data.email)
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )
        try:
            hashed_password = get_password_hash(user_create_data.password)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        created_user: User = await self.users_repository.create_user(
            db,
            user_create_data=user_create_data,
            hashed_password=hashed_password,
        )
        email_verification_token_payload = {
            "sub": str(created_user.id),
            "email": created_user.email,
            "is_admin": created_user.is_admin,
        }
        email_verification_token = create_jwt_token(email_verification_token_payload, "SECRET_JWT_EMAIL", expires=1440) # expires in 1 day
        email_verification_link = f"http://localhost:4200/verify-account?token={email_verification_token}"

        await send_registration_email(user_create_data.email, email_verification_link, conf)

        return created_user

    async def get_user_by_id(self, db: AsyncSession, user_id: int) -> User:
        user_to_fetch: User | None = await self.users_repository.get_by_id(db, user_id=user_id)

        if user_to_fetch is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User to fetch does not exist"
            )

        return user_to_fetch

    async def verify_account(self, db: AsyncSession, token: str) -> User:
        decoded_jwt_token: dict | None = decode_jwt_token(token, "SECRET_JWT_EMAIL")
        if decoded_jwt_token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired or is invalid"
            )

        if decoded_jwt_token.get("email") is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token: email not found in token payload"
            )

        email:str = decoded_jwt_token["email"]
        id:int = int(decoded_jwt_token["sub"])

        user_to_verify: User | None = await self.users_repository.get_by_id(db, user_id=id)

        if user_to_verify is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User to verify does not exist"
            )
        if user_to_verify.email != email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token: email in token does not match user's email"
            )
        if user_to_verify.is_account_verified:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Account is already verified"
            )

        try:
            user_to_verify.is_account_verified = True
            await self.users_repository.save_changes(db)
            await db.refresh(user_to_verify)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error verifying account: {str(e)}"
            ) from e

        return user_to_verify

    async def resend_verification_email(self, db: AsyncSession, token: str, conf: ConnectionConfig) -> User:
        decoded_jwt_token: dict = decode_jwt_token(token, "SECRET_JWT_EMAIL")

        decoded_jwt_token: dict = decode_jwt_token_without_exp_check(token, "SECRET_JWT_EMAIL")

        email: str = decoded_jwt_token["email"]
        id: int = int(decoded_jwt_token["sub"])

        user_to_verify: User | None = await self.users_repository.get_by_id(db, user_id=id)

        if user_to_verify is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User to verify does not exist"
            )
        if user_to_verify.email != email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token: email in token does not match user's email"
            )
        if user_to_verify.is_account_verified:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Account is already verified"
            )

        email_verification_token_payload = {
            "sub": str(user_to_verify.id),
            "email": user_to_verify.email,
            "is_admin": user_to_verify.is_admin,
        }
        email_verification_token = create_jwt_token(email_verification_token_payload, "SECRET_JWT_EMAIL", expires=1440) # expires in 1 day
        email_verification_link = f"http://localhost:4200/verify-account?token={email_verification_token}"

        await send_registration_email(user_to_verify.email, email_verification_link, conf)

        return user_to_verify