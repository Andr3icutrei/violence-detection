from pydantic import BaseModel, EmailStr
from typing import Optional

class UserBase(BaseModel):
    email: EmailStr
    credits: int = 0
    is_active: bool = True
    is_admin: bool = False
    auth_provider: Optional[str] = None

class CreateUserDto(UserBase):
    password: str

class UpdateUserDto(UserBase):
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    credits: Optional[int] = None

class UserResponseDto(UserBase):
    id: int

    class Config:
        from_attributes = True