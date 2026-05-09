import cv2
import tempfile
import os

from fastapi import HTTPException
from starlette import status
from ultralytics.trackers.basetrack import BaseTrack


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
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output.name
    output.close()

    output_fps = fps if fps > 0 else 24.0
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), output_fps, (width, height))
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


def run_people_tracking(temp_video_path: str, yolo_model) -> tuple[str, int]:
    BaseTrack.reset_id()
    if hasattr(yolo_model, 'predictor') and yolo_model.predictor is not None:
        if hasattr(yolo_model.predictor, 'trackers'):
            for tracker in yolo_model.predictor.trackers:
                tracker.reset()
    tracked_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tracked_video_path = tracked_temp_file.name
    tracked_temp_file.close()

    cap = cv2.VideoCapture(temp_video_path)
    writer = None
    final_output_path = ""
    tracked_ids = set()
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

        writer = cv2.VideoWriter(
            tracked_video_path,
            cv2.VideoWriter_fourcc(*"avc1"),
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

        final_output_path = tracked_video_path
        return tracked_video_path, len(tracked_ids)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if os.path.exists(tracked_video_path) and tracked_video_path != final_output_path:
            os.remove(tracked_video_path)