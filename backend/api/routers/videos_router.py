from fastapi import APIRouter

router = APIRouter(
    prefix="/videos",
    tags=["Videos"],
)

videos_service = VideosService()
