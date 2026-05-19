import os
import tempfile

from fastapi import HTTPException
from starlette import status

from helpers.bucket_helper import download_object_to_file, download_model_to_file
from helpers.inference_helper import run_classification_and_occlusion, run_classification_only
from schemas.classification import ClassificationResponseDto
from services.inference_runtime import InferenceRuntime


class ClassificationService:
    def __init__(self, inference_runtime: InferenceRuntime):
        self.inference_runtime = inference_runtime

    async def classify_and_occlusion_video(
        self,
        video_path: str,
        inference_model_path: str,
        inference_model_kind: str | None = None,
    ) -> tuple[ClassificationResponseDto, str]:
        temp_video_path = ""
        temp_model_path = ""
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
            model_suffix = os.path.splitext(inference_model_path)[1] or ".onnx"
            temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix=model_suffix)
            temp_model_path = temp_model_file.name
            temp_model_file.close()
            was_model_downloaded = await download_model_to_file(inference_model_path, temp_model_path)
            if not was_model_downloaded:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Inference model not found in storage."
                )
            overlay_video_path, predicted_label, confidence, predicted_class_probability = run_classification_and_occlusion(
                temp_model_path,
                temp_video_path,
                self.inference_runtime,
                inference_model_kind=inference_model_kind,
                inference_model_cache_key=inference_model_path,
            )
            if temp_model_path and os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            result = ClassificationResponseDto(
                video_path=overlay_video_path,
                predicted_label=str(predicted_label),
                confidence=str(confidence),
                predicted_class_probability=str(predicted_class_probability),
            )
            return result, temp_video_path
        except Exception as e:
            for path in (overlay_video_path, temp_video_path, temp_model_path):
                if path and os.path.exists(path):
                    os.remove(path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred during classification: {str(e)}"
            )

    async def classify_video(
        self,
        video_path: str,
        inference_model_path: str,
        inference_model_kind: str | None = None,
    ) -> ClassificationResponseDto:
        temp_video_path = ""
        temp_model_path = ""
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

            model_suffix = os.path.splitext(inference_model_path)[1] or ".onnx"
            temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix=model_suffix)
            temp_model_path = temp_model_file.name
            temp_model_file.close()

            was_model_downloaded = await download_model_to_file(inference_model_path, temp_model_path)
            if not was_model_downloaded:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Inference model not found in storage."
                )

            predicted_label, confidence, predicted_class_probability = run_classification_only(
                temp_model_path,
                temp_video_path,
                self.inference_runtime,
                inference_model_kind=inference_model_kind,
                inference_model_cache_key=inference_model_path,
            )

            if temp_model_path and os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            result = ClassificationResponseDto(
                video_path="",
                predicted_label=str(predicted_label),
                confidence=str(confidence),
                predicted_class_probability=str(predicted_class_probability),
            )
            return result

        except Exception as e:
            for path in (temp_video_path, temp_model_path):
                if path and os.path.exists(path):
                    os.remove(path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred during classification: {str(e)}"
            )
