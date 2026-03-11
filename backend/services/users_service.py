from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from schemas.users_schema import CreateUserDto
from core.security import get_password_hash
from repositories.users_repository import UsersRepository
from models.user import User

class UsersService:
    def __init__(self):
        self.users_repository = UsersRepository()

    def create_user(self, db: Session, user_create_data: CreateUserDto) -> User:
        db_user:User = self.users_repository.get_by_email(db, email=user_create_data.email)

        if db_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )

        hashed_password = get_password_hash(user_create_data.password)
        return self.users_repository.create_user(db, email=user_create_data.email, hashed_password=hashed_password)

    def get_user_by_id(self, db: Session, user_id: int) -> User:
        user_to_fetch:User = self.users_repository.get_by_id(db, user_id=user_id)

        if user_to_fetch is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User to fetch does not exist"
            )

        return user_to_fetch
