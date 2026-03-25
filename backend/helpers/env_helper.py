import os
from dotenv import load_dotenv

def get_env_variable(name: str, default: str | None = None) -> str:
    load_dotenv()
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable '{name}' is not set and no default value provided.")
    return value