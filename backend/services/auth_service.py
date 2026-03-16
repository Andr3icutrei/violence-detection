from fastapi import Depends, HTTPException, status
import jwt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os

from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_token_from_cookie
from core.database import get_db
from models.user import User
from repositories.users_repository import UsersRepository
from core.security import verify_password

class AuthService:
    def __init__(self):
        self.users_repository: UsersRepository = UsersRepository()

    async def login(self, db: AsyncSession, email: str, password: str):
        user: User | None = await self.users_repository.get_by_email(db, email)

        if user is None:
            raise HTTPException(
                status_code = status.HTTP_404_NOT_FOUND,
                detail = f"User with email {email} not found."
            )

        if verify_password(password, user.hashed_password):
            return user
        else:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail = "Incorrect password."
            )

    @staticmethod
    def create_jwt_token(data: dict) -> str:
        to_encode = data.copy()

        expire = datetime.now(timezone.utc) + timedelta(minutes=60)
        to_encode.update({"exp": expire})

        load_dotenv()
        SECRET_JWT_KEY:str = os.getenv("SECRET_JWT_KEY")
        ALGORITHM:str = os.getenv("JWT_ALGORITHM")

        encoded_jwt:str = jwt.encode(to_encode, SECRET_JWT_KEY, ALGORITHM)
        return encoded_jwt

    @staticmethod
    def _decode_jwt_token(token: str) -> dict:
        load_dotenv()

        SECRET_JWT_KEY:str = os.getenv("SECRET_JWT_KEY")
        ALGORITHM:str = os.getenv("JWT_ALGORITHM")

        try:
            decoded_token = jwt.decode(token, SECRET_JWT_KEY, algorithms=[ALGORITHM])
            return decoded_token if decoded_token["exp"] >= datetime.now(timezone.utc).timestamp() else None
        except jwt.PyJWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    @staticmethod
    async def get_current_user(
        token: str = Depends(get_token_from_cookie),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        decoded_token = AuthService._decode_jwt_token(token)

        if decoded_token is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")

        user_id = decoded_token.get("sub")

        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        try:
            parsed_user_id = int(user_id)
        except (TypeError, ValueError):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        user: User | None = await UsersRepository().get_by_id(db, parsed_user_id)

        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        return user

    @staticmethod
    async def get_current_admin_user(
        token: str = Depends(get_token_from_cookie),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        decoded_token = AuthService._decode_jwt_token(token)

        if decoded_token is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")

        user_id = decoded_token.get("sub")

        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        try:
            parsed_user_id = int(user_id)
        except (TypeError, ValueError):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        user: User | None = await UsersRepository().get_by_id(db, parsed_user_id)

        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        if user.is_admin is False:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have admin privileges")

        return user