from typing import List

from pydantic import BaseModel

from models.action import Action

class InferenceActionResponseDto(BaseModel):
    id: int
    name: str
    action_id: Action
    credits: int

class UpdateInferenceActionRequestDto(BaseModel):
    id: int
    new_credits: int

class PatchInferenceActionRequestDto(BaseModel):
    actions: List[UpdateInferenceActionRequestDto]
