from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

from schemas.users_schema import CreateUserDto
from core.security import get_password_hash
from repositories.users_repository import UsersRepository
from models.user import User

class UsersService:
    def __init__(self):
        self.users_repository = UsersRepository()

    async def create_user(self, db: AsyncSession, user_create_data: CreateUserDto) -> User:
        db_user: User | None = await self.users_repository.get_by_email(db, email=user_create_data.email)

        if db_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )

        hashed_password = get_password_hash(user_create_data.password)
        return await self.users_repository.create_user(
            db,
            user_create_data=user_create_data,
            hashed_password=hashed_password,
        )

    async def get_user_by_id(self, db: AsyncSession, user_id: int) -> User:
        user_to_fetch: User | None = await self.users_repository.get_by_id(db, user_id=user_id)

        if user_to_fetch is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User to fetch does not exist"
            )

        return user_to_fetch
