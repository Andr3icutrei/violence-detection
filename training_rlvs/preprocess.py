import os
import cv2
import json
import random
import argparse
import numpy as np
import heapq
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_keyframes(video_path: str, n_frames: int = 16) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    top_frames = []
    frame_idx = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        diff = cv2.absdiff(curr_frame, prev_frame)
        score = float(np.mean(diff))
        if len(top_frames) < n_frames:
            heapq.heappush(top_frames, (score, frame_idx, curr_frame.copy()))
        else:
            heapq.heappushpop(top_frames, (score, frame_idx, curr_frame.copy()))
        prev_frame = curr_frame
        frame_idx += 1
    cap.release()
    if not top_frames:
        return []
    top_frames.sort(key=lambda x: x[1])
    return [item[2] for item in top_frames]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

def collect_video_paths(directory: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(os.path.join(root, fname))
    return sorted(paths)

def process_subset(video_paths: list[str], labels: list[int], frames_dir: Path, n_frames: int, subset_name: str) -> list[dict]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    skipped = 0

    def process_single_video(video_path, label):
        frames = extract_keyframes(video_path, n_frames=n_frames)
        if not frames:
            return None
        video_stem = Path(video_path).stem
        local_manifest = []
        for frame_idx, frame in enumerate(frames):
            filename = f"{subset_name}_{video_stem}_{frame_idx:02d}.jpg"
            save_path = frames_dir / filename
            is_success, im_buf_arr = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if is_success:
                im_buf_arr.tofile(str(save_path))
            local_manifest.append({
                "frame_path": str(save_path),
                "label": label,
                "source_video": video_path,
                "frame_index": frame_idx,
            })
        return local_manifest

    completed_count = 0
    total_videos = len(video_paths)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_video, vp, lbl): (vp, lbl) for vp, lbl in zip(video_paths, labels)}
        for future in as_completed(futures):
            result = future.result()
            completed_count += 1
            if result is None:
                vp = futures[future][0]
                print(f"  [WARN] Skipped: {vp}")
                skipped += 1
            else:
                manifest.extend(result)
            if completed_count % 50 == 0 or completed_count == total_videos:
                print(f"  [{subset_name}] {completed_count}/{total_videos} -> {len(manifest)} (skipped: {skipped})")
    return manifest

def preprocess(violence_dir: str, nonviolence_dir: str, output_dir: str, n_frames: int = 16, val_ratio: float = 0.10, test_ratio: float = 0.25, seed: int = 42) -> None:
    random.seed(seed)
    output_path = Path(output_dir)
    print("\n[1/4] Colectare...")
    violence_paths = collect_video_paths(violence_dir)
    nonviolence_paths = collect_video_paths(nonviolence_dir)
    if not violence_paths or not nonviolence_paths:
        raise FileNotFoundError("Nu am gasit videouri")

    print("\n[2/4] Split...")
    def split_class(paths: list[str]) -> tuple[list, list, list]:
        paths_shuffled = paths.copy()
        random.shuffle(paths_shuffled)
        n = len(paths_shuffled)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test = paths_shuffled[:n_test]
        val = paths_shuffled[n_test:n_test + n_val]
        train = paths_shuffled[n_test + n_val:]
        return train, val, test

    v_train, v_val, v_test = split_class(violence_paths)
    nv_train, nv_val, nv_test = split_class(nonviolence_paths)

    def combine(pos, neg):
        paths = pos + neg
        labels = [1] * len(pos) + [0] * len(neg)
        combined = list(zip(paths, labels))
        random.shuffle(combined)
        p, l = zip(*combined) if combined else ([], [])
        return list(p), list(l)

    train_paths, train_labels = combine(v_train, nv_train)
    val_paths, val_labels = combine(v_val, nv_val)
    test_paths, test_labels = combine(v_test, nv_test)

    print("\n[3/4] Extragere...")
    start_time = datetime.now()
    subsets = [("train", train_paths, train_labels), ("val", val_paths, val_labels), ("test", test_paths, test_labels)]
    manifests = {}
    for subset_name, paths, labels in subsets:
        frames_dir = output_path / subset_name / "frames"
        manifest = process_subset(paths, labels, frames_dir, n_frames, subset_name)
        manifests[subset_name] = manifest
        manifest_path = output_path / f"{subset_name}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n[4/4] Statistici...")
    def count_labels(manifest):
        violence = sum(1 for e in manifest if e["label"] == 1)
        nonviolence = sum(1 for e in manifest if e["label"] == 0)
        return {"total": len(manifest), "violence": violence, "nonviolence": nonviolence}

    stats = {
        "created_at": datetime.now().isoformat(),
        "n_frames": n_frames,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "elapsed_sec": round(elapsed, 1),
        "violence_dir": violence_dir,
        "nonviolence_dir": nonviolence_dir,
        "subsets": {name: count_labels(manifests[name]) for name in ("train", "val", "test")},
    }

    stats_path = output_path / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nFinalizat in {elapsed:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--violence_dir", required=True)
    parser.add_argument("--nonviolence_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocess(
        violence_dir=args.violence_dir,
        nonviolence_dir=args.nonviolence_dir,
        output_dir=args.output_dir,
        n_frames=args.n_frames,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )