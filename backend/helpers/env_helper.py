import os
from functools import lru_cache
from dotenv import load_dotenv

@lru_cache()
def load_env_once():
    load_dotenv()

def get_env_variable(name: str, default: str | None = None) -> str:
    load_env_once()
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable '{name}' is not set and no default value provided.")
    return value