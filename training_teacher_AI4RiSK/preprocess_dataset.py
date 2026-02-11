import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from cv2 import dnn_superres

# ================= CONFIGURARE =================
INPUT_ROOT = Path(r"../../Datasets/AI4RiSK")
OUTPUT_ROOT = Path(r"../../Datasets/AI4RiSK_CROPPED_SR")
TARGET_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.3
PADDING_PIXELS = 10
# Calea catre modelul descarcat (FSRCNN_x3.pb)
SR_MODEL_PATH = "fsrcnn_x3.pb"
USE_SUPER_RES = True  # Seteaza pe False daca vrei doar Lanczos (mai rapid, calitate mai slaba)


# ===============================================

def init_super_res():
    """Initializare model Super Resolution"""
    if not os.path.exists(SR_MODEL_PATH):
        print(f"ATENTIE: Nu am gasit {SR_MODEL_PATH}. Voi folosi doar resize standard.")
        return None

    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(SR_MODEL_PATH)
        # Setam modelul si scara (x3 inseamna ca inmulteste rezolutia cu 3)
        sr.setModel("fsrcnn", 3)
        # Daca vrei pe GPU (CUDA):
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Model Super Resolution incarcat cu succes pe GPU.")
        return sr
    except Exception as e:
        print(f"Eroare la incarcarea SR: {e}")
        return None


def get_person_bbox_yolo(frames, model):
    if not frames: return None

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    found_person = False

    # Pentru viteza, putem verifica 1 din 2 frame-uri, dar pentru precizie lasam toate
    for frame in frames:
        results = model.predict(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False, device=0)

        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                found_person = True
                coords = boxes.xyxy.cpu().numpy()
                for box in coords:
                    x1, y1, x2, y2 = box
                    min_x = min(min_x, x1);
                    min_y = min(min_y, y1)
                    max_x = max(max_x, x2);
                    max_y = max(max_y, y2)

    if not found_person:
        h, w = frames[0].shape[:2]
        return (0, 0, w, h)

    h_img, w_img = frames[0].shape[:2]
    x1 = int(max(0, min_x - PADDING_PIXELS))
    y1 = int(max(0, min_y - PADDING_PIXELS))
    x2 = int(min(w_img, max_x + PADDING_PIXELS))
    y2 = int(min(h_img, max_y + PADDING_PIXELS))

    # Asiguram dimensiuni valide
    if (x2 - x1) <= 0 or (y2 - y1) <= 0: return (0, 0, w_img, h_img)

    return (x1, y1, x2 - x1, y2 - y1)


def process_frame_smart(frame, bbox, target_size, sr_model=None):
    x, y, w, h = bbox
    crop = frame[y:y + h, x:x + w]

    if crop.size == 0: return cv2.resize(frame, target_size)

    # 1. Aplicam Super Resolution daca crop-ul e mic
    # Daca crop-ul e deja mai mare decat target-ul, nu are sens sa facem SR
    if sr_model and (w < target_size[0] or h < target_size[1]):
        try:
            # Upscale (ex: de la 50x50 -> 150x150)
            crop = sr_model.upsample(crop)
        except:
            pass  # Fallback la resize normal in caz de eroare

    # 2. Square Padding (Benzi negre)
    h_c, w_c = crop.shape[:2]
    max_dim = max(h_c, w_c)

    top = (max_dim - h_c) // 2
    bottom = max_dim - h_c - top
    left = (max_dim - w_c) // 2
    right = max_dim - w_c - left

    padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 3. Final Resize cu LANCZOS4 (mult mai clar decat LINEAR)
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_LANCZOS4)

    return resized


def main():
    print("Incarcare modele...")
    yolo_model = YOLO("yolov8n.pt")
    sr_model = init_super_res() if USE_SUPER_RES else None

    input_path_obj = Path(INPUT_ROOT)
    all_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mpg']:
        all_files.extend(input_path_obj.rglob(ext))

    print(f"Start procesare {len(all_files)} videouri...")

    count = 0
    for file_p in all_files:
        try:
            rel_path = file_p.relative_to(input_path_obj)
            out_p = Path(OUTPUT_ROOT) / rel_path
            out_p.parent.mkdir(parents=True, exist_ok=True)

            # Citire
            cap = cv2.VideoCapture(str(file_p))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            cap.release()

            if len(frames) < 5: continue

            # Detectie BBox Globala
            bbox = get_person_bbox_yolo(frames, yolo_model)

            # Scriere
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_p), fourcc, 25.0, TARGET_SIZE)

            for frame in frames:
                # Aici apelam functia noua de procesare
                processed = process_frame_smart(frame, bbox, TARGET_SIZE, sr_model)
                out.write(processed)
            out.release()

            count += 1
            if count % 10 == 0:  # Printam mai des
                print(f"Procesat: {count}/{len(all_files)}")

        except Exception as e:
            print(f"Eroare {file_p}: {e}")

    print("Finalizat.")


if __name__ == '__main__':
    main()