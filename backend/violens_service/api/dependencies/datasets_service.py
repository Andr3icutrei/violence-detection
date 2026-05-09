from fastapi import Depends

from api.dependencies.datasets_repository import get_datasets_repository
from api.dependencies.datasets_updated_ws import get_datasets_updated_ws
from api.dependencies.users_repository import get_users_repository
from api.dependencies.videos_repository import get_videos_repository
from api.routers.datasets_ws_router import DatasetUpdatedWs
from repositories.datasets_repository import DatasetsRepository
from repositories.users_repository import UsersRepository
from repositories.videos_repository import VideosRepository
from services.datasets_service import DatasetsService


def get_datasets_service(
    dataset_updates_ws: DatasetUpdatedWs = Depends(get_datasets_updated_ws),
    datasets_repository: DatasetsRepository = Depends(get_datasets_repository),
    users_repository: UsersRepository = Depends(get_users_repository),
    videos_repository: VideosRepository = Depends(get_videos_repository),
) -> DatasetsService:
    return DatasetsService(
        datasets_repository=datasets_repository,
        users_repository=users_repository,
        videos_repository=videos_repository,
        notifier=dataset_updates_ws,
    )

