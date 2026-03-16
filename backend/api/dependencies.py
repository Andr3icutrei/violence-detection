from fastapi import HTTPException, Request, status

def get_token_from_cookie(request: Request) -> str:
	token = request.cookies.get("access_token")
	if not token:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Not authenticated",
		)
	return token
