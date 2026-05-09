from fastapi import HTTPException, Request, status

ACCESS_TOKEN_COOKIE = "access_token"


def get_token_from_cookie(request: Request) -> str:
    token = request.cookies.get(ACCESS_TOKEN_COOKIE)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )
    return token

