from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from repositories.users_repository import UsersRepository


def get_users_repository(db: AsyncSession = Depends(get_db)) -> UsersRepository:
    return UsersRepository(db)

