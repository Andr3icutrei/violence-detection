import os
import tempfile

from fastapi import HTTPException
from starlette import status

from helpers.bucket_helper import download_object_to_file
from shared_models import InferenceModel

from helpers.inference_helper import run_classification_and_gradcam
from schemas.classification import ClassificationResponseDto
from services.inference_runtime import InferenceRuntime


class ClassificationService:
    def __init__(self, inference_runtime: InferenceRuntime):
        self.inference_runtime = inference_runtime

    async def classify_and_gradcam_video(
        self,
        video_path: str,
        inference_model: InferenceModel
    ) -> tuple[ClassificationResponseDto, str]:
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
            overlay_video_path, predicted_label, confidence, predicted_class_probability = run_classification_and_gradcam(
                inference_model,
                temp_video_path,
                self.inference_runtime
            )
            result = ClassificationResponseDto(
                video_path=overlay_video_path,
                predicted_label=str(predicted_label),
                confidence=str(confidence),
                predicted_class_probability=str(predicted_class_probability),
            )
            return result, temp_video_path
        except Exception as e:
            for path in (overlay_video_path, temp_video_path):
                if path and os.path.exists(path):
                    os.remove(path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred during classification: {str(e)}"
            )