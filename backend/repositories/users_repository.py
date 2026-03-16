from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from models.user import User
from schemas.users_schema import CreateUserDto

class UsersRepository:
    async def get_by_id(self, db: AsyncSession, user_id: int) -> User | None:
        result = await db.execute(select(User).filter(User.id == user_id))
        return result.scalars().first()

    async def get_by_email(self, db: AsyncSession, email: str) -> User | None:
        result = await db.execute(select(User).filter(User.email == email))
        return result.scalars().first()

    async def create_user(self, db: AsyncSession, user_create_data: CreateUserDto, hashed_password: str) -> User:
        try:
            user = User(
                email=user_create_data.email,
                hashed_password=hashed_password,
                credits=user_create_data.credits,
                is_active=user_create_data.is_active,
                is_admin=user_create_data.is_admin,
                auth_provider=user_create_data.auth_provider
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user
        except SQLAlchemyError as e:
            await db.rollback()
            raise e

    async def get_all(self, db: AsyncSession) -> list[User]:
        result = await db.execute(select(User))
        return list(result.scalars().all())
