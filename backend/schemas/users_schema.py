from pydantic import BaseModel, EmailStr, ConfigDict, Field
from typing import Optional

class UserBase(BaseModel):
    email: EmailStr
    credits: int = 0
    is_active: bool = True
    is_admin: bool = False
    auth_provider: Optional[str] = None

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

    class Config:
        from_attributes = True