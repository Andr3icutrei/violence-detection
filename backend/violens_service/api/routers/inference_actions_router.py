from typing import List

from fastapi import APIRouter, Depends, HTTPException
from starlette import status

from api.dependencies import get_inference_actions_service
from models import User
from models.inference_action import InferenceAction
from schemas.inference_actions_schema import InferenceActionResponseDto, PatchInferenceActionRequestDto
from services.auth_service import get_current_user, get_current_admin_user
from services.inference_actions_service import InferenceActionsService

router = APIRouter(
    prefix="/inference_actions",
    tags=["Inference Actions"],
)

@router.get("/get_inference_actions_for_dataset/{dataset_id}", response_model=List[InferenceActionResponseDto], status_code=status.HTTP_200_OK)
async def get_inference_actions_for_dataset(
    dataset_id: int,
    user: User = Depends(get_current_user),
    inference_actions_service: InferenceActionsService = Depends(get_inference_actions_service),
):
    result: List[InferenceAction] = await inference_actions_service.get_inference_actions_for_dataset(dataset_id=dataset_id)
    return [
        InferenceActionResponseDto(
            id=action.id,
            name=action.action_id.name.replace("_", " ").title(),
            action_id=action.action_id,
            credits=action.credits,
        ) for action in result
    ]

@router.get("/get_inference_actions_stats", response_model=List[InferenceActionResponseDto], status_code=status.HTTP_200_OK)
async def get_inference_actions_stats(
    user: User = Depends(get_current_user),
    inference_actions_service: InferenceActionsService = Depends(get_inference_actions_service),
):
    result: List[InferenceAction] = await inference_actions_service.get_inference_actions_stats()
    return [
        InferenceActionResponseDto(
            id=action.id,
            name=action.action_id.name.replace("_", " ").title(),
            action_id=action.action_id,
            credits=action.credits,
        ) for action in result
    ]

@router.patch("/update_credits_inference_actions",  status_code=status.HTTP_200_OK)
async def update_credits_for_action(
    actions_to_patch: PatchInferenceActionRequestDto,
    current_admin_user = Depends(get_current_admin_user),
    inference_actions_service: InferenceActionsService = Depends(get_inference_actions_service),
):
    await inference_actions_service.update_credits_for_action(actions_to_patch)
