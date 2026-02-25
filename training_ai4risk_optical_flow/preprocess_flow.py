import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- CONFIGURARE ---
INPUT_DATASET_ROOT = Path("../../Datasets/AI4RiSK_CROPPED_SR_V2")
OUTPUT_DATASET_ROOT = Path("../../Datasets/AI4RiSK_FLOW")

# Parametrii Optical Flow
FLOW_BOUND = 20.0
TARGET_SIZE = 224
USE_FLOAT16 = True  # True pentru a economisi 50% spatiu pe disk

# Numarul de procese paralele.
# De obicei: numarul de nuclee fizice sau logice.
# Poti seta manual (ex: 8) sau lasa automat.
NUM_WORKERS = multiprocessing.cpu_count() - 4


def compute_optical_flow(frames, flow_bound):
    """
    Calculeaza Dense Optical Flow (Farneback).
    Aceasta functie ruleaza in interiorul fiecarui proces worker.
    """
    flows = []
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    for i in range(len(gray_frames) - 1):
        prev = gray_frames[i]
        curr = gray_frames[i + 1]

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        flow = np.clip(flow, -flow_bound, flow_bound)
        flow = (flow + flow_bound) / (2 * flow_bound)
        flows.append(flow)

    if len(flows) > 0:
        flows.append(flows[-1])
    else:
        h, w = gray_frames[0].shape
        flows.append(np.full((h, w, 2), 0.5, dtype=np.float32))

    return np.stack(flows, axis=0)


def process_single_video(video_path):
    """
    Functie wrapper care proceseaza un singur video.
    Returneaza 1 daca succes, 0 daca eroare/skip (pentru counter).
    """
    try:
        # Reconstruim caile (trebuie facut aici pt ca Path objects uneori fac probleme la pickling)
        video_path = Path(video_path)
        rel_path = video_path.relative_to(INPUT_DATASET_ROOT)
        output_path = OUTPUT_DATASET_ROOT / rel_path.with_suffix('.npy')

        if output_path.exists():
            return 0  # Skip

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Resize preventiv daca e cazul
            if frame.shape[0] != TARGET_SIZE or frame.shape[1] != TARGET_SIZE:
                frame = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))

            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            return 0

        # Calcul intensiv
        flow_data = compute_optical_flow(frames, FLOW_BOUND)

        if USE_FLOAT16:
            flow_data = flow_data.astype(np.float16)
        else:
            flow_data = flow_data.astype(np.float32)

        np.save(str(output_path), flow_data)
        return 1

    except Exception as e:
        # Printam eroarea dar nu oprim tot procesul
        print(f"\nError processing {video_path}: {e}")
        return 0


def main():
    # 1. Colectare fisiere
    print(f"Scanning videos in {INPUT_DATASET_ROOT}...")
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mpg']
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(list(INPUT_DATASET_ROOT.rglob(ext)))

    # Convertim la stringuri pentru siguranta multiprocessing
    all_videos_str = [str(p) for p in all_videos]

    total_files = len(all_videos_str)
    print(f"Found {total_files} videos.")
    print(f"Starting processing with {NUM_WORKERS} parallel workers...")
    print(f"Output: {OUTPUT_DATASET_ROOT}")

    # 2. Procesare Paralela
    processed_count = 0

    # ProcessPoolExecutor gestioneaza automat workerii
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Trimitem toate joburile
        futures = [executor.submit(process_single_video, vid) for vid in all_videos_str]

        # Bara de progres care se actualizeaza pe masura ce workerii termina
        for future in tqdm(as_completed(futures), total=total_files, unit="vid"):
            result = future.result()
            processed_count += result

    print(f"\nProcessing complete! Successfully processed {processed_count}/{total_files} new videos.")


if __name__ == "__main__":
    # Aceasta protectie este obligatorie pe Windows pentru multiprocessing
    main()