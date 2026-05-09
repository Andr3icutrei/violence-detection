from google.auth.transport import requests
from fastapi import Depends, HTTPException, status
from google.oauth2 import id_token

from api.dependencies.token_from_cookie import get_token_from_cookie
from api.dependencies.users_repository import get_users_repository
from core.security import verify_password
from helpers.env_helper import get_env_variable
from helpers.jwt_helper import decode_jwt_token
from models.user import User
from repositories.users_repository import UsersRepository
from schemas.auth_schema import TokenSchema

class AuthService:
    def __init__(self, users_repository: UsersRepository):
        self.users_repository: UsersRepository = users_repository

    async def login(self, email: str, password: str):
        user: User | None = await self.users_repository.get_by_email(email)
        if user is None:
            raise HTTPException(
                status_code = status.HTTP_404_NOT_FOUND,
                detail = f"User with email {email} not found."
            )
        if not user.is_account_verified:
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
        if verify_password(password, user.hashed_password):
            return user
        else:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = "Incorrect password."
            )

    async def login_google(self, data: TokenSchema):
        try:
            id_info = id_token.verify_oauth2_token(data.tokenId, requests.Request(), get_env_variable("GOOGLE_AUTH_CLIENT_ID"))
            email = id_info.get("email")
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token")

        if email is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email not found in token")

        user: User | None = await self.users_repository.get_by_email(email)

        if user is None:
            try:
                user = await self.users_repository.create_user_google(email)
            except Exception as e:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error creating user: {str(e)}")
        else:
            if user.is_banned:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error_code": "ACCOUNT_BANNED",
                        "message": "Account is banned. Please contact support for more information.",
                    },
                )
            if user.auth_provider == "local":
                try:
                    user.auth_provider = "google"
                    await self.users_repository.save_changes(user)
                except Exception as e:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error updating user authentication provider: {str(e)}")
        return user


async def get_current_user(
    token: str = Depends(get_token_from_cookie),
    users_repository: UsersRepository = Depends(get_users_repository)
) -> User:
    decoded_token = decode_jwt_token(token, "SECRET_JWT_KEY")

    if decoded_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")

    user_id = decoded_token.get("sub")

    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    try:
        parsed_user_id = int(user_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user: User | None = await users_repository.get_by_id(parsed_user_id)

    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return user


async def get_current_admin_user(
    token: str = Depends(get_token_from_cookie),
    users_repository: UsersRepository = Depends(get_users_repository)
) -> User:
    decoded_token = decode_jwt_token(token, "SECRET_JWT_KEY")

    if decoded_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")

    user_id = decoded_token.get("sub")

    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    try:
        parsed_user_id = int(user_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user: User | None = await users_repository.get_by_id(parsed_user_id)

    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have admin privileges")

    return user