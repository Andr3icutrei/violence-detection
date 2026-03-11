from sqlalchemy.orm import Session
from models.user import User
from schemas.users_schema import CreateUserDto

class UsersRepository:
    def get_by_id(self, db: Session, user_id: int) -> User | None:
        return db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, db: Session, email: str) -> User | None:
        return db.query(User).filter(User.email == email).first()

    def create_user(self, db: Session, user_create_data: CreateUserDto, hashed_password: str) -> User:
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
            db.commit()
            db.refresh(user)
            return user
        except Exception as e:
            db.rollback()
            raise e

    def get_all(self, db: Session) -> list[User]:
        return db.query(User).all()