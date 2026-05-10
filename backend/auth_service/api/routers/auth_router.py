from fastapi import APIRouter


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

@router.get("/verify")
async def verify():
