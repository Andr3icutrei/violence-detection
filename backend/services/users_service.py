from typing import List

from jwt import PyJWTError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status
from fastapi_mail import ConnectionConfig
import jwt

from helpers.env_helper import get_env_variable
from schemas.users_schema import CreateUserDto, UserResponseDto
from core.security import get_password_hash
from repositories.users_repository import UsersRepository
from models.user import User
from helpers.email_helper import send_registration_email, send_reset_password_email, send_user_ban_email
from helpers.jwt_helper import create_jwt_token, decode_jwt_token, decode_jwt_token_without_exp_check, decode_jwt_token_reset_password

class UsersService:
    def __init__(self):
        self.users_repository = UsersRepository()

    async def create_user(self, user_create_data: CreateUserDto, conf: ConnectionConfig, db: AsyncSession) -> User:
        db_user: User | None = await self.users_repository.get_by_email(user_create_data.email, db)
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
            user_create_data=user_create_data,
            hashed_password=hashed_password,
            db=db,
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

    async def get_user_by_id(self, user_id: int, db: AsyncSession) -> User:
        user_to_fetch: User | None = await self.users_repository.get_by_id(user_id=user_id, db=db)

        if user_to_fetch is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User to fetch does not exist"
            )

        return user_to_fetch

    async def verify_account(self, token: str, db: AsyncSession) -> User:
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

        user_to_verify: User | None = await self.users_repository.get_by_id(user_id=id, db=db)

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

    async def resend_verification_email(self, token: str, conf: ConnectionConfig, db: AsyncSession) -> User:
        decoded_jwt_token: dict = decode_jwt_token_without_exp_check(token, "SECRET_JWT_EMAIL")

        email: str = decoded_jwt_token["email"]
        id: int = int(decoded_jwt_token["sub"])

        user_to_verify: User | None = await self.users_repository.get_by_id(user_id=id, db=db)

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

    async def send_reset_password_email(self, email: str, conf: ConnectionConfig, db: AsyncSession) -> User:
        user: User | None = await self.users_repository.get_by_email(email=email, db=db)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User with the provided email does not exist"
            )
        if user.is_account_verified is False:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "ACCOUNT_NOT_VERIFIED",
                    "message": "Account not verified. Please check your email for the verification link.",
                },
            )
        if user.is_banned:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "ACCOUNT_BANNED",
                    "message": "Account is banned. Please contact support for more information.",
                },
            )
        reset_password_token_payload = {
            "sub": str(user.id),
            "email": user.email,
        }
        dynamic_secret = f"SECRET_JWT_RESET_PASSWORD_{user.hashed_password}"
        reset_password_token = create_jwt_token(
            reset_password_token_payload,
            dynamic_secret,
            expires=15
        )
        reset_password_link = f"http://localhost:4200/reset-password?token={reset_password_token}"
        await send_reset_password_email(user.email, reset_password_link, conf)
        return user

    async def reset_password(self, token: str, new_password: str, db: AsyncSession) -> User:
        try:
            unverified_payload = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token format"
            )
        email = unverified_payload.get("email")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token payload is missing the email"
            )
        user_to_reset_password: User | None = await self.users_repository.get_by_email(email=email, db=db)
        if user_to_reset_password is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User with the provided email does not exist"
            )
        decoded_jwt_token: dict = decode_jwt_token_reset_password(token, f"SECRET_JWT_RESET_PASSWORD_{user_to_reset_password.hashed_password}")
        try:
            user_to_reset_password.hashed_password = get_password_hash(new_password)
            await self.users_repository.save_changes(db)
            await db.refresh(user_to_reset_password)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error resetting password: {str(e)}"
            ) from e
        return user_to_reset_password

    async def verify_reset_password_token(self, token: str, db: AsyncSession) -> bool:
        try:
            unverified_payload = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
        except PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token format"
            )
        email = unverified_payload.get("email")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token payload is missing the email"
            )
        user: User | None = await self.users_repository.get_by_email(email=email, db=db)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User with the provided email does not exist"
            )
        try:
            decode_jwt_token_reset_password(token, f"SECRET_JWT_RESET_PASSWORD_{user.hashed_password}")
            return True
        except PyJWTError:
            return False

    async def update_all_users_credits(self, db: AsyncSession) -> None:
        try:
            credits_to_update = int(get_env_variable("DEFAULT_CREDITS"))
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error loading environment variable DEFAULT_CREDITS: {str(e)}"
            ) from e
        users: List[User] = await self.users_repository.get_all(db)
        await self.users_repository.update_users_credits(users, credits_to_update, db)

    async def get_all_users(self,
        search_term: str | None,
        page: int,
        page_size: int,
        db: AsyncSession,
    ) -> List[UserResponseDto]:
        users: List[User] = await self.users_repository.get_all(db)

        if search_term:
            search_term_lower = search_term.lower()
            users = [user for user in users if search_term_lower in user.email.lower()]

        start_index = page * page_size
        end_index = start_index + page_size
        paginated_users = users[start_index:end_index]

        return [
            UserResponseDto(
                id=user.id,
                email=user.email,
                credits=user.credits,
                is_admin=user.is_admin,

            )
            for user in paginated_users
        ]

    async def update_user_role(self, user_id: int, is_admin: bool, db: AsyncSession) -> None:
        user_to_update: User | None = await self.users_repository.get_by_id(user_id=user_id, db=db)
        if user_to_update is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User to update does not exist"
            )
        try:
            user_to_update.is_admin = is_admin
            await self.users_repository.save_changes(db)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error updating user role: {str(e)}"
            ) from e

    async def ban_user(self, user_id: int, reason: str, conf: ConnectionConfig, db: AsyncSession) -> None:
        user_to_ban: User | None = await self.users_repository.get_by_id(user_id=user_id, db=db)
        if user_to_ban is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User to ban does not exist"
            )
        try:
            user_to_ban.is_banned = True
            user_to_ban.ban_reason = reason
            await self.users_repository.save_changes(db)
            await send_user_ban_email(user_to_ban.email, reason, conf=conf)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error banning user: {str(e)}"
            ) from e