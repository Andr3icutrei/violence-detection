import cv2
import tempfile
import os

from fastapi import HTTPException
from starlette import status
from ultralytics.trackers.basetrack import BaseTrack


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

    import numpy as np

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
            
        # Probe write to ensure internal encoder works
        probe_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        writer.write(probe_frame)
        
        # Test if it flushed properly or check file size. Python cv2.VideoWriter.write returns None usually,
        # but re-opening allows checking if file was created successfully.
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

    output_fps = fps if fps > 0 else 24.0
    output_path, writer = _create_temp_writer(output_fps, (width, height))
    if not writer.isOpened():
        raise ValueError("Could not initialize video writer for Grad-CAM output.")

    if total_source_frames <= 0:
        total_source_frames = len(overlays)

    num_overlays = len(overlays)
    
    import numpy as np
    
    # Pre-process overlays
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
            # Assume matplotlib RBG -> BGR conversion
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

    return output_path


def run_people_tracking(temp_video_path: str, yolo_model) -> tuple[str, int]:
    import subprocess

    BaseTrack.reset_id()
    if hasattr(yolo_model, 'predictor') and yolo_model.predictor is not None:
        if hasattr(yolo_model.predictor, 'trackers'):
            for tracker in yolo_model.predictor.trackers:
                tracker.reset()

    cap = cv2.VideoCapture(temp_video_path)
    writer = None
    final_output_path = ""
    tracked_ids = set()
    tracked_video_path = ""

    try:
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not open downloaded video for tracking."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid video dimensions for tracking output."
            )

        tracked_video_path, writer = _create_temp_writer(
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not initialize video writer for tracking output.",
            )

        written_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model.track(
                source=frame,
                persist=True,
                classes=[0],
                conf=0.25,
                tracker="botsort.yaml",
                verbose=False,
            )

            tracked_frame = frame.copy()
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
                xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []

                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = [int(v) for v in box[:4]]
                    track_id = ids[i] if i < len(ids) else -1
                    if track_id != -1:
                        tracked_ids.add(track_id)

                    cv2.rectangle(tracked_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(
                        tracked_frame,
                        f"ID {track_id}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        1,
                    )

            writer.write(tracked_frame)
            written_frames += 1

        if written_frames == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tracking output video contains zero frames.",
            )

        writer.release()
        writer = None

        if not os.path.exists(tracked_video_path) or os.path.getsize(tracked_video_path) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tracking output video file is missing or empty.",
            )

        h264_output_path = tracked_video_path + "_h264.mp4"

        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", tracked_video_path,
                "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
                "-pix_fmt", "yuv420p", h264_output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if os.path.exists(tracked_video_path):
                os.remove(tracked_video_path)

            final_output_path = h264_output_path
        except Exception:
            final_output_path = tracked_video_path

        return final_output_path, len(tracked_ids)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if tracked_video_path and os.path.exists(tracked_video_path) and tracked_video_path != final_output_path:
            os.remove(tracked_video_path)