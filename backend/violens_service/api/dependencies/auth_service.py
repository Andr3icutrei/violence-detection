from fastapi import Depends

from api.dependencies.users_repository import get_users_repository
from repositories.users_repository import UsersRepository
from services.auth_service import AuthService


def get_auth_service(
    users_repository: UsersRepository = Depends(get_users_repository),
) -> AuthService:
    return AuthService(users_repository)

