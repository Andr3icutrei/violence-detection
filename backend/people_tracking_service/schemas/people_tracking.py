from pydantic import BaseModel


class PeopleTrackingResponseDto(BaseModel):
    people_tracked: str
    video_path: str