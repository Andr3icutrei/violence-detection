from pydantic import BaseModel


class DatasetResponseDto(BaseModel):
    id: int
    name: str