from typing import List

from pydantic import BaseModel

from models.action import Action

class InferenceActionResponseDto(BaseModel):
    id: int
    name: str
    action_id: Action
    credits: int
