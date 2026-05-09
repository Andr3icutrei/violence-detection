from pydantic import BaseModel


class ClassificationResponseDto(BaseModel):
    predicted_label: str
    confidence: str
    predicted_class_probability: str
    video_path: str