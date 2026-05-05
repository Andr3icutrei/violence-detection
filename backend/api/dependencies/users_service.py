from fastapi import Depends

from api.dependencies.users_repository import get_users_repository
from api.routers.users_ws_router import UserUpdatedWs, get_users_updated
from repositories.users_repository import UsersRepository
from services.users_service import UsersService


def get_users_service(
    users_repository: UsersRepository = Depends(get_users_repository),
    users_updated_ws: UserUpdatedWs = Depends(get_users_updated),
) -> UsersService:
    return UsersService(
        users_repository=users_repository,
        notifier=users_updated_ws,
    )

