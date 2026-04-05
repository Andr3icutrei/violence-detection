from pydantic import BaseModel

class VideoResponseDto(BaseModel):
    id: int
    uid: str
    name: str
    path: str
    dataset_id: int
    dataset_name: str
    is_violent: bool
    duration: int
    frame_rate: int