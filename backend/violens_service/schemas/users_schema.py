from pydantic import BaseModel, EmailStr, ConfigDict, Field
from typing import Optional, List


class UserBase(BaseModel):
    email: EmailStr
    credits: int = 0
    is_active: bool = True
    is_admin: bool = False
    is_banned: bool = False
    ban_reason: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class CreateUserDto(UserBase):
    password: str = Field(min_length=1, max_length=72)

class UpdateUserDto(UserBase):
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(default=None, min_length=1, max_length=72)
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    credits: Optional[int] = None

class UserResponseDto(UserBase):
    id: int
    credits_used: Optional[int] = None
    class Config:
        from_attributes = True

class TopbarInformationDto(BaseModel):
    email: EmailStr
    credits: int
    nameInitials: str

class UserBanRequestDto(BaseModel):
    ban_reason: str

class UsersStatsResponseDto(BaseModel):
    active_users: int
    inactive_users: int
    banned_users: int
    most_active_users: List[UserResponseDto]