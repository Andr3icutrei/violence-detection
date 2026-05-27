import importlib

_DEPENDENCY_MODULES = {
    "get_auth_service": "api.dependencies.auth_service",
    "get_credits_service": "api.dependencies.credits_service",
    "get_datasets_repository": "api.dependencies.datasets_repository",
    "get_datasets_service": "api.dependencies.datasets_service",
    "get_datasets_updated_ws": "api.dependencies.datasets_updated_ws",
    "get_inference_actions_repository": "api.dependencies.inference_actions_repository",
    "get_inference_actions_service": "api.dependencies.inference_actions_service",
    "get_inference_history_repository": "api.dependencies.inference_history_repository",
    "get_inference_history_service": "api.dependencies.inference_history_service",
    "get_inference_models_repository": "api.dependencies.inference_models_repository",
    "get_token_from_cookie": "api.dependencies.token_from_cookie",
    "get_users_repository": "api.dependencies.users_repository",
    "get_users_service": "api.dependencies.users_service",
    "get_videos_repository": "api.dependencies.videos_repository",
    "get_videos_service": "api.dependencies.videos_service",
}


__all__ = list(_DEPENDENCY_MODULES.keys())


def __getattr__(name: str):
    module_path = _DEPENDENCY_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module 'api.dependencies' has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)
