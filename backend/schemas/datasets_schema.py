from pydantic import BaseModel

from schemas.users_schema import UserResponseDto

class DatasetResponseDto(BaseModel):
    id: int
    name: str
    is_official: bool
    user: UserResponseDto | None = None