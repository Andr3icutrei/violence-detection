import os
from typing import List, Dict

from dotenv import load_dotenv
from fastapi import HTTPException, status
from sqlalchemy import select, update, bindparam, func, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from models import InferenceHistory
from models.user import User
from schemas.users_schema import CreateUserDto

class UsersRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: int) -> User | None:
        result = await self.db.execute(select(User).filter(User.id == user_id))
        return result.scalars().first()

    async def get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(select(User).filter(User.email == email))
        return result.scalars().first()

    async def create_user(self, user_create_data: CreateUserDto, hashed_password: str) -> User:
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
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)
            return user
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise e

    async def create_user_google(self, email: str) -> User:
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
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)
            return user
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise e

    async def get_all_exclude_banned(self) -> List[User]:
        result = await self.db.execute(select(User).where(User.is_banned == False))
        return list(result.scalars().all())

    async def get_users_paged(self, search_term: str | None, page: int, page_size: int) -> List[User]:
        query = select(User).where(User.is_banned == False)
        if search_term:
            query = query.where(User.email.ilike(f"%{search_term}%"))
        query = query.offset(page * page_size).limit(page_size)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def save_changes(self, user: User) -> None:
        try:
            await self.db.commit()
            await self.db.refresh(user)
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise e

    async def add_user(self, user: User) -> User:
        try:
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)
            return user
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise e

    async def update_users_credits(self, users: List[User], credits_to_update: int) -> List[User]:
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
        try:
            await self.db.execute(stmt, update_data)
            await self.db.commit()
            for user in users:
                await self.db.refresh(user)
        except SQLAlchemyError as e:
            await self.db.rollback()
            raise e
        return users

    async def count_active_users(self) -> int:
        query = select(func.count()).select_from(User).where(
            User.is_active == True,
            User.is_banned == False
        )
        result = await self.db.execute(query)
        return result.scalar_one()

    async def count_inactive_users(self) -> int:
        query = select(func.count()).select_from(User).where(
            or_(
                User.is_active == False,
                User.is_banned == False
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one()

    async def count_banned_users(self) -> int:
        query = select(func.count()).select_from(User).where(
            User.is_banned == True
        )
        result = await self.db.execute(query)
        return result.scalar_one()

    async def get_most_active_users(self) -> Dict[User, int]:
        credits_sum = func.coalesce(func.sum(InferenceHistory.credits_used), 0)
        stmt = (
            select(User, credits_sum)
                .outerjoin(InferenceHistory, User.id == InferenceHistory.user_id)
                .group_by(User.id)
                .order_by(credits_sum.desc())
                .limit(3)
        )
        result = await self.db.execute(stmt)
        return {user: total_credits for user, total_credits in result.all()}