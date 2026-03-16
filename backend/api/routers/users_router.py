from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from schemas.users_schema import CreateUserDto, UserResponseDto
from services.users_service import UsersService

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)

users_service = UsersService()

@router.post("/create", response_model = UserResponseDto, status_code=status.HTTP_201_CREATED)
async def create_user(user_create_data: CreateUserDto, db: AsyncSession = Depends(get_db)):
    return await users_service.create_user(db, user_create_data)

@router.get("/{user_id}", response_model=UserResponseDto, status_code=status.HTTP_200_OK)
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    return await users_service.get_user_by_id(db, user_id)
