from fastapi import Depends

from api.dependencies.inference_actions_repository import get_inference_actions_repository
from api.dependencies.inference_history_repository import get_inference_history_repository
from api.dependencies.inference_runtime import get_inference_runtime
from api.dependencies.users_repository import get_users_repository
from api.dependencies.videos_repository import get_videos_repository
from repositories.inference_actions_repository import InferenceActionsRepository
from repositories.inference_history_repository import InferenceHistoryRepository
from repositories.users_repository import UsersRepository
from repositories.videos_repository import VideosRepository
from services.inference_runtime import InferenceRuntime
from services.videos_service import VideosService


def get_videos_service(
    inference_runtime: InferenceRuntime = Depends(get_inference_runtime),
    videos_repository: VideosRepository = Depends(get_videos_repository),
    inference_history_repository: InferenceHistoryRepository = Depends(get_inference_history_repository),
    users_repository: UsersRepository = Depends(get_users_repository),
    inference_actions_repository: InferenceActionsRepository = Depends(get_inference_actions_repository),
) -> VideosService:
    return VideosService(
        inference_runtime=inference_runtime,
        videos_repository=videos_repository,
        inference_history_repository=inference_history_repository,
        users_repository=users_repository,
        inference_actions_repository=inference_actions_repository,
    )

