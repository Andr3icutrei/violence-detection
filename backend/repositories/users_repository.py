import os
from typing import List

from dotenv import load_dotenv
from fastapi import HTTPException, status
from sqlalchemy import select, update, bindparam
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
            try:
                load_dotenv()
                DEFAULT_CREDITS:int = int(os.getenv("DEFAULT_CREDITS"))
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error loading environment variables: {str(e)}"
                )
            user = User(
                email=user_create_data.email,
                hashed_password=hashed_password,
                credits=DEFAULT_CREDITS,
                is_active=True,
                is_account_verified=False,
                is_admin=user_create_data.is_admin,
                auth_provider="local"
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user
        except SQLAlchemyError as e:
            await db.rollback()
            raise e

    async def create_user_google(self, db: AsyncSession, email: str) -> User:
        try:
            try:
                load_dotenv()
                DEFAULT_CREDITS:int = int(os.getenv("DEFAULT_CREDITS"))
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error loading environment variables: {str(e)}"
                )
            user = User(
                email=email,
                hashed_password="",
                credits=DEFAULT_CREDITS,
                is_active=True,
                is_account_verified=True,
                is_admin=False,
                auth_provider="google"
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user
        except SQLAlchemyError as e:
            await db.rollback()
            raise e

    async def get_all(self, db: AsyncSession) -> List[User]:
        result = await db.execute(select(User))
        return list(result.scalars().all())

    async def save_changes(self, db: AsyncSession) -> None:
        try:
            await db.commit()
        except SQLAlchemyError as e:
            await db.rollback()
            raise e

    async def add_user(self, db: AsyncSession, user: User) -> User:
        try:
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user
        except SQLAlchemyError as e:
            await db.rollback()
            raise e

    async def update_users_credits(self, db: AsyncSession, users: List[User], credits_to_update: int) -> List[User]:
        if not users:
            return users
        stmt = (
            update(User)
            .where(User.id == bindparam("b_id"))
            .values(credits=bindparam("new_credits"))
        )
        update_data = [
            {"b_id": user.id, "new_credits": user.credits + credits_to_update} 
            for user in users
        ]
        await db.execute(stmt, update_data)
        await db.commit()
        for user in users:
            await db.refresh(user)
        return users

