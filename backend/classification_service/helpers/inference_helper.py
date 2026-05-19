import cv2
import tempfile
import os
from fastapi import HTTPException
from starlette import status

from inference.onnx.pipeline import Onnx3dInferencePipeline
from inference.onnx.preprocess import preprocess_video_for_inference, prepare_input_tensors

def run_classification_and_gradcam(
    inference_model_path: str,
    temp_video_path: str,
    inference_runtime,
    inference_model_kind: str | None = None,
    inference_model_cache_key: str | None = None,
) -> tuple[str, bool, float, float]:
    overlay_video_path = ""
    try:
        pipeline = Onnx3dInferencePipeline(inference_model_path)
        frames, frames_by_name = preprocess_video_for_inference(
            temp_video_path,
            inference_runtime.yolo_model,
            pipeline.input_specs,
        )
        if not frames:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not prepare input frames for inference.",
            )

        input_tensors = prepare_input_tensors(frames_by_name, pipeline.input_specs)
        pred_class, probs, heatmap = pipeline.predict_and_generate_cam(input_tensors)
        confidence = float(max(probs))
        predicted_class_probability = float(probs[pred_class])
        overlays = pipeline.overlay_heatmap_on_frames(frames, heatmap)

        overlay_video_path = write_overlay_video(
            overlays,
            temp_video_path,
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
    import subprocess
    import numpy as np
    import shutil

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
            ("MJPG", ".avi"),
            ("XVID", ".avi"),
            ("avc1", ".mp4"),
        ]
        for codec, suffix in candidates:
            output = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            output_path = output.name
            output.close()
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, size)

            if not writer.isOpened():
                writer.release()
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue

            probe_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            writer.write(probe_frame)

            writer.release()
            if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                os.remove(output_path)
                continue

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
    print(f"DEBUG: S-a deschis video-ul. FPS={fps}, Frame-uri totale={total_source_frames}", flush=True)
    ret, test_frame = cap.read()
    print(f"DEBUG: Primul frame a putut fi citit? {ret}", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    output_path, writer = _create_temp_writer(output_fps, (width, height))
    if not writer.isOpened():
        raise ValueError("Could not initialize video writer for Grad-CAM output.")

    if total_source_frames <= 0:
        total_source_frames = len(overlays)

    num_overlays = len(overlays)

    processed_overlays = []
    for frame in overlays:
        if frame is None:
            processed_overlays.append(None)
            continue

        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        processed_overlays.append(frame)

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

    final_output_path = output_path + "_h264.mp4"

    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_path and not os.path.exists(ffmpeg_path):
        ffmpeg_path = None
    if not ffmpeg_path:
        ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")

    if not ffmpeg_path:
        raise ValueError(
            "ffmpeg was not found. Install ffmpeg or set FFMPEG_PATH to ffmpeg.exe to "
            "enable H.264 MP4 output for browser playback."
        )

    try:
        subprocess.run([
            ffmpeg_path, "-y", "-i", output_path,
            "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p", final_output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(output_path):
            os.remove(output_path)
        return final_output_path

    except Exception as e:
        print(f"Eroare la conversia ffmpeg: {e}", flush=True)
        return output_path