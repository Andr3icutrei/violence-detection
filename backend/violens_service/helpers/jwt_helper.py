import os
from datetime import timezone, datetime, timedelta
from typing import Literal

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException
from starlette import status

def create_jwt_token(data: dict, jwt_key: Literal["SECRET_JWT_KEY", "SECRET_JWT_EMAIL"] | str, expires: int) -> str:
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(minutes=expires)
    to_encode.update({"exp": expire})

    load_dotenv()
    secret_key: str = os.getenv(jwt_key)
    
    if secret_key is None:
        secret_key = jwt_key
        
    ALGORITHM: str = os.getenv("JWT_ALGORITHM")

    encoded_jwt: str = jwt.encode(to_encode, secret_key, ALGORITHM)
    return encoded_jwt


def decode_jwt_token(token: str, jwt_key: Literal["SECRET_JWT_KEY", "SECRET_JWT_EMAIL"]) -> dict:
    load_dotenv()
    SECRET_JWT_KEY: str = os.getenv(jwt_key)
    ALGORITHM: str = os.getenv("JWT_ALGORITHM")

    try:
        decoded_token = jwt.decode(token, SECRET_JWT_KEY, algorithms=[ALGORITHM])
        return decoded_token if decoded_token["exp"] >= datetime.now(timezone.utc).timestamp() else None
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def decode_jwt_token_without_exp_check(token: str, jwt_key: Literal["SECRET_JWT_KEY", "SECRET_JWT_EMAIL"]) -> dict:
    load_dotenv()
    SECRET_JWT_KEY: str = os.getenv(jwt_key)
    ALGORITHM: str = os.getenv("JWT_ALGORITHM")

    try:
        decoded_token = jwt.decode(token, SECRET_JWT_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        return decoded_token
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def decode_jwt_token_reset_password(token: str, jwt_key: str) -> dict:
    load_dotenv()
    ALGORITHM: str = os.getenv("JWT_ALGORITHM")
    try:
        decoded_token = jwt.decode(token, jwt_key, algorithms=[ALGORITHM])
        return decoded_token if decoded_token["exp"] >= datetime.now(timezone.utc).timestamp() else None
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")