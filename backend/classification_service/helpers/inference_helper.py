import cv2
import tempfile
import os
from fastapi import HTTPException
from starlette import status

from inference.slowfast.pipeline import prepare_slowfast_tensors
from inference.slowfast.preprocess import preprocess_video_for_inference as preprocess_slowfast_video
from inference.resnet3d.pipeline import prepare_r3d_tensor
from inference.resnet3d.preprocess import preprocess_video_for_inference as preprocess_resnet3d_video
from shared_models import InferenceModel

def run_classification_and_gradcam(
    inference_model: InferenceModel,
    temp_video_path: str,
    inference_runtime
) -> tuple[str, bool, float, float]:
    overlay_video_path = ""
    try:
        if inference_model is InferenceModel.SLOWFAST_NETWORK:
            config = inference_runtime.slowfast_config
            frames = preprocess_slowfast_video(temp_video_path, inference_runtime.yolo_model_path, config)
            if not frames:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not prepare input frames for inference.",
                )
            slow_tensor, fast_tensor = prepare_slowfast_tensors(frames, config)
            pred_class, probs, heatmap = inference_runtime.slowfast_pipeline.predict_and_generate_cam(
                slow_tensor,
                fast_tensor,
            )
            confidence = float(max(probs))
            predicted_class_probability = float(probs[pred_class])
            overlays = inference_runtime.slowfast_pipeline.overlay_heatmap_on_frames(frames, heatmap)
        elif inference_model is InferenceModel.RESNET3D_NETWORK:
            config = inference_runtime.resnet3d_config
            frames = preprocess_resnet3d_video(temp_video_path, inference_runtime.yolo_model, config)
            if not frames:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not prepare input frames for inference.",
                )

            input_tensor = prepare_r3d_tensor(frames, config)
            pred_class, probs, heatmap = inference_runtime.resnet3d_pipeline.predict_and_generate_cam(input_tensor)
            confidence = float(max(probs))
            predicted_class_probability = float(probs[pred_class])
            overlays = inference_runtime.resnet3d_pipeline.overlay_heatmap_on_frames(frames, heatmap)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unsupported inference model configured for this video's dataset."
            )
        overlay_video_path = write_overlay_video(
            overlays,
            temp_video_path
        )
        return overlay_video_path, pred_class, confidence, predicted_class_probability
    except HTTPException:
        if overlay_video_path and os.path.exists(overlay_video_path):
            os.remove(overlay_video_path)
        raise
    except Exception as exc:
        if overlay_video_path and os.path.exists(overlay_video_path):
            os.remove(overlay_video_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build Grad-CAM video output: {exc}",
        )

def write_overlay_video(overlays: list, source_video_path: str) -> str:
    if not overlays:
        raise ValueError("No Grad-CAM overlays were produced for this video.")

    first_frame = overlays[0]
    if first_frame is None or len(first_frame.shape) < 2:
        raise ValueError("Invalid overlay frame format.")

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise ValueError("Could not open source video to determine properties.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    height, width = first_frame.shape[:2]
    
    def _create_temp_writer(fps: float, size: tuple[int, int]) -> tuple[str, cv2.VideoWriter]:
        # Prefer software-friendly AVI codecs; allow override via env.
        forced_codec = os.getenv("VIDEO_OUTPUT_CODEC")
        forced_ext = os.getenv("VIDEO_OUTPUT_EXT")
        if forced_codec:
            suffix = forced_ext if forced_ext and forced_ext.startswith(".") else ".avi"
            output = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            output_path = output.name
            output.close()
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*forced_codec), fps, size)
            return output_path, writer

        candidates = [
            ("mp4v", ".mp4"),
            ("vp80", ".webm"),
            ("avc1", ".mp4"),
            ("MJPG", ".avi"),
            ("XVID", ".avi"),
        ]
        for codec, suffix in candidates:
            output = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            output_path = output.name
            output.close()
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, size)
            if writer.isOpened():
                return output_path, writer
            writer.release()
            if os.path.exists(output_path):
                os.remove(output_path)

        output = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        output_path = output.name
        output.close()
        return output_path, cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)

    output_fps = fps if fps > 0 else 24.0
    output_path, writer = _create_temp_writer(output_fps, (width, height))
    if not writer.isOpened():
        raise ValueError("Could not initialize video writer for Grad-CAM output.")

    if total_source_frames <= 0:
        total_source_frames = len(overlays)

    num_overlays = len(overlays)
    
    # Pre-process overlays
    processed_overlays = []
    for frame in overlays:
        if frame is None:
            processed_overlays.append(None)
            continue
        
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
        if frame.dtype != "uint8":
            frame = frame.clip(0, 255).astype("uint8")
            
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
            
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        processed_overlays.append(bgr_frame)

    written_frames = 0
    frame_idx = 0

    while True:
        ret, _ = cap.read()
        if not ret:
            break

        idx = int((frame_idx / total_source_frames) * num_overlays)
        if idx >= num_overlays:
            idx = num_overlays - 1

        overlay_frame = processed_overlays[idx]
        if overlay_frame is not None:
            writer.write(overlay_frame)
            written_frames += 1

        frame_idx += 1

    cap.release()
    writer.release()

    if written_frames == 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise ValueError("Grad-CAM output video contains zero writable frames.")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise ValueError("Grad-CAM output file is empty after encoding.")

    return output_path
