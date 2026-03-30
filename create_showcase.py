import cv2
import numpy as np
import os
import random
from pathlib import Path

# Paths to the datasets
DATASET_PATH = Path("../Datasets")
DATASETS = ["AI4RiSK"]

def get_random_video(dataset_name, label):
    """ label should be 'Violence' or 'NonViolence' """
    video_dir = DATASET_PATH / dataset_name / label
    if not video_dir.exists():
        return None
    videos = [str(p) for p in video_dir.rglob("*.*") if p.suffix.lower() in [".avi", ".mp4", ".mpeg"]]
    if not videos:
        return None
    return random.choice(videos)

def read_video_frames(video_path, num_frames=30, size=(640, 480)):
    """ Reads a fixed number of evenly spaced frames from a video. """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return [np.zeros((size[1], size[0], 3), dtype=np.uint8)] * num_frames

    step = max(total_frames // num_frames, 1)
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * step, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, size)
            frames.append(frame)
        else:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    cap.release()
    return frames

def main():
    print("Selecting random videos...")
    
    # Pick one of each type in sequence
    videos_to_fetch = [
        ("AI4RiSK", "0"),
        ("AI4RiSK", "1"),
        ("AI4RiSK", "0"),
        ("AI4RiSK", "2"),
        ("AI4RiSK", "0"),
        ("AI4RiSK", "3"),
        ("AI4RiSK", "0"),
        ("AI4RiSK", "4"),
        ("AI4RiSK", "0"),
        ("AI4RiSK", "1"),
        ("AI4RiSK", "0"),
        ("AI4RiSK", "2"),
        ("AI4RiSK", "0"),
        ("AI4RiSK", "3"),
        ("AI4RiSK", "0"),
        ("AI4RiSK", "4")
    ]
    
    all_frames = []
    for ds, label in videos_to_fetch:
        v_path = get_random_video(ds, label)
        if v_path:
            print(f"Reading frames from {v_path}...")
            # Get 30 frames per video (approx 2 seconds per clip at 15fps)
            frames = read_video_frames(v_path, num_frames=30, size=(640, 480))
            all_frames.extend(frames)
        else:
            print(f"Could not find video for {ds} - {label}")

    if not all_frames:
        print("No frames could be gathered.")
        return

    out_path = "dataset_showcase.webm"
    print(f"Generating sequence of {len(all_frames)} frames...")
    
    try:
        import imageio
        gif_path = "dataset_showcase.gif"
        images = []
        for frame in all_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(rgb_frame)
        imageio.mimsave(gif_path, images, fps=15, loop=0)
        print(f"Successfully saved {gif_path}")
    except ImportError:
        print("imageio not found. Saving as WebM video instead. You can install imageio with 'pip install imageio' for GIF support.")
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(out_path, fourcc, 15.0, (640, 480))
        for frame in all_frames:
            out.write(frame)
        out.release()
        print(f"Successfully saved {out_path}")

if __name__ == "__main__":
    main()
