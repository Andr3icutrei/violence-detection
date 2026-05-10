import os
import tempfile

from fastapi import HTTPException
from fastapi.params import Depends
from starlette import status

from core.dependencies.inference_runtime import get_inference_runtime
from helpers.bucket_helper import download_object_to_file

from helpers.inference_helper import run_people_tracking
from services.inference_runtime import InferenceRuntime


class PeopleTrackingService:
    def __init__(self, inference_runtime: InferenceRuntime = Depends(get_inference_runtime)):
        self.inference_runtime = inference_runtime

    async def people_tracking(self, video_path: str) -> tuple[str, int, str]:
        temp_video_path = ""
        overlay_video_path = ""
        try:
            temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video_path = temp_video_file.name
            temp_video_file.close()
            was_downloaded = await download_object_to_file(video_path, temp_video_path)
            if not was_downloaded:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Video file not found in storage."
                )
            overlay_video_path, people_tracked = run_people_tracking(
                temp_video_path,
                self.inference_runtime.yolo_model
            )
            return overlay_video_path, people_tracked, temp_video_path
        except Exception as e:
            for path in (overlay_video_path, temp_video_path):
                if path and os.path.exists(path):
                    os.remove(path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred during people tracking: {str(e)}"
            )