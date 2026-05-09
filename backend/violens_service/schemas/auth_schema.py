from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional

class LoginResponseDto(BaseModel):
    access_token: str
    token_type: str = "bearer"
    model_config = ConfigDict(from_attributes=True)

class LoginRequestDto(BaseModel):
    email: EmailStr
    password: str
    model_config = ConfigDict(from_attributes=True)

class LogoutResponseDto(BaseModel):
    message: str = "Successfully logged out"
    model_config = ConfigDict(from_attributes=True)

class TokenSchema(BaseModel):
    tokenId: str