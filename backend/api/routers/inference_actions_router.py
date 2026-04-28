from typing import List

from fastapi import APIRouter
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from core.database import get_db
from models import User
from models.action import Action
from models.inference_action import InferenceAction
from schemas.inference_actions_schema import InferenceActionResponseDto
from services.auth_service import AuthService
from services.inference_actions_service import InferenceActionsService

router = APIRouter(
    prefix="/inference_actions",
    tags=["Inference Actions"],
)

auth_service = AuthService()
inference_actions_service = InferenceActionsService()

@router.get("/get_inference_actions_for_dataset/{dataset_id}", response_model=List[InferenceActionResponseDto], status_code=status.HTTP_200_OK)
async def get_inference_actions_for_dataset(
    dataset_id: int,
    user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result: List[InferenceAction] = await inference_actions_service.get_inference_actions_for_dataset(dataset_id=dataset_id, db=db)
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
    user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result: List[InferenceAction] = await inference_actions_service.get_inference_actions_stats(db=db)
    return [
        InferenceActionResponseDto(
            id=action.id,
            name=action.action_id.name.replace("_", " ").title(),
            action_id=action.action_id,
            credits=action.credits,
        ) for action in result
    ]

@router.patch("/update_credits_for_action", response_model=InferenceActionResponseDto, status_code=status.HTTP_200_OK)
async def update_credits_for_action(
    action_id: Action,
    credits: int,
    current_admin_user = Depends(auth_service.get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    inference_action_updated: InferenceAction = await inference_actions_service.update_credits_for_action(action_id, credits, db)
    return InferenceActionResponseDto(
        id=inference_action_updated.id,
        name=inference_action_updated.action_id.name.replace("_", " ").title(),
        action_id=inference_action_updated.action_id,
        credits=inference_action_updated.credits,
    )