import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# CONSTANTELE TALE
# ==========================================
TARGET_SIZE = (224, 224)
MIN_CROP_SIZE = (140, 140)
CONFIDENCE_THRESHOLD = 0.25
PADDING_PIXELS = 20


# ==========================================
# FUNCȚIA DE CROP FINAL (PROCESARE)
# ==========================================
def process_frame_final(frame, bbox, target_size):
    x, y, w, h = bbox
    crop = frame[y:y + h, x:x + w]

    # Fallback dacă din vreo eroare cutia este goală
    if crop.size == 0:
        return cv2.resize(frame, target_size)

    h_c, w_c = crop.shape[:2]
    max_dim = max(h_c, w_c)

    # Calculăm padding-ul necesar pentru a face imaginea perfect pătrată (fără a o deforma)
    top = (max_dim - h_c) // 2
    bottom = max_dim - h_c - top
    left = (max_dim - w_c) // 2
    right = max_dim - w_c - left

    padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(padded, target_size, interpolation=cv2.INTER_LANCZOS4)


# ==========================================
# FUNCȚIA PRINCIPALĂ DE DEBUG
# ==========================================
def save_detailed_preprocessing_steps(video_path, yolo_model_path):
    print(f"Încărcare model YOLO din: {yolo_model_path}")
    model = YOLO(yolo_model_path)

    print(f"Deschidere video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        print("Eroare: Video prea scurt pentru a fi procesat.")
        return

    h_img, w_img = frames[0].shape[:2]

    # ==========================================
    # ETAPA 1: YOLO DETECTION (din 10 în 10 cadre)
    # ==========================================
    print("Etapa 1: Rulare YOLO...")
    yolo_min_x, yolo_min_y = float('inf'), float('inf')
    yolo_max_x, yolo_max_y = float('-inf'), float('-inf')
    found_person_yolo = False

    yolo_demo_frame = None
    yolo_demo_idx = -1  # Salvăm indexul cadrului demonstrativ

    for i in range(0, len(frames), 10):
        # Rulăm predicția pe cadrul curent
        results = model.predict(frames[i], classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                found_person_yolo = True

                # Salvăm cadrul de demo prima dată când detectăm pe cineva
                if yolo_demo_idx == -1:
                    yolo_demo_idx = i
                    yolo_demo_frame = frames[i].copy()

                coords = boxes.xyxy.cpu().numpy()
                for box in coords:
                    x1, y1, x2, y2 = box

                    # Desenăm TOATE persoanele detectate (Portocaliu) pe cadrul de demo
                    if i == yolo_demo_idx:
                        cv2.rectangle(yolo_demo_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)

                    # Acumulăm coordonatele maxime și minime pentru Global BBox
                    yolo_min_x, yolo_min_y = min(yolo_min_x, x1), min(yolo_min_y, y1)
                    yolo_max_x, yolo_max_y = max(yolo_max_x, x2), max(yolo_max_y, y2)

    bbox_yolo = None
    if found_person_yolo:
        # Aplicăm padding-ul de 20px și ne asigurăm că nu ieșim din limitele imaginii
        x1 = int(max(0, yolo_min_x - PADDING_PIXELS))
        y1 = int(max(0, yolo_min_y - PADDING_PIXELS))
        x2 = int(min(w_img, yolo_max_x + PADDING_PIXELS))
        y2 = int(min(h_img, yolo_max_y + PADDING_PIXELS))
        bbox_yolo = (x1, y1, x2 - x1, y2 - y1)

        # Desenăm cutia YOLO Globală (Roșu)
        cv2.rectangle(yolo_demo_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(yolo_demo_frame, "YOLO Global BBox", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite("1_yolo_detection.jpg", yolo_demo_frame)
        print("Salvat: 1_yolo_detection.jpg")

    # ==========================================
    # ETAPA 2: MOTION DETECTION (din 3 în 3 cadre)
    # ==========================================
    print("Etapa 2: Rulare Motion Detection...")
    # Primul cadru - blurat
    first_gray = cv2.GaussianBlur(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), (21, 21), 0)
    cv2.imwrite("2a_motion_first_frame_gaussian.jpg", first_gray)
    print("Salvat: 2a_motion_first_frame_gaussian.jpg")

    mot_min_x, mot_min_y = float('inf'), float('inf')
    mot_max_x, mot_max_y = float('-inf'), float('-inf')
    found_motion = False
    motion_demo_idx = 1

    # Căutăm cadrul demonstrativ de mișcare și calculăm coordonatele globale
    for i in range(1, len(frames), 3):
        gray = cv2.GaussianBlur(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), (21, 21), 0)
        frame_delta = cv2.absdiff(first_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_has_valid_motion = False
        for c in cnts:
            if cv2.contourArea(c) < 150:
                continue

            found_motion = True
            frame_has_valid_motion = True
            (x, y, w, h) = cv2.boundingRect(c)

            # Acumulăm pentru Motion Global BBox
            mot_min_x, mot_min_y = min(mot_min_x, x), min(mot_min_y, y)
            mot_max_x, mot_max_y = max(mot_max_x, x + w), max(mot_max_y, y + h)

        # Păstrăm primul frame cu mișcare validă pentru pozele de debug
        if frame_has_valid_motion and motion_demo_idx == 1:
            motion_demo_idx = i

    bbox_motion = None
    if found_motion:
        # Aplicăm padding-ul de 20px
        x1 = int(max(0, mot_min_x - PADDING_PIXELS))
        y1 = int(max(0, mot_min_y - PADDING_PIXELS))
        x2 = int(min(w_img, mot_max_x + PADDING_PIXELS))
        y2 = int(min(h_img, mot_max_y + PADDING_PIXELS))
        bbox_motion = (x1, y1, x2 - x1, y2 - y1)

        # Recreăm pașii doar pentru frame-ul demonstrativ ca să le salvăm pe disc
        demo_gray = cv2.GaussianBlur(cv2.cvtColor(frames[motion_demo_idx], cv2.COLOR_BGR2GRAY), (21, 21), 0)
        cv2.imwrite("2b_motion_current_gaussian.jpg", demo_gray)

        demo_delta = cv2.absdiff(first_gray, demo_gray)
        cv2.imwrite("2c_motion_abs_diff.jpg", demo_delta)

        demo_thresh = cv2.dilate(cv2.threshold(demo_delta, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        cv2.imwrite("2d_motion_threshold_dilated.jpg", demo_thresh)

        # Desenăm contururile individuale și BBox-ul Global pe cadrul demonstrativ
        demo_contours_img = frames[motion_demo_idx].copy()
        cnts, _ = cv2.findContours(demo_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) >= 150:
                cv2.drawContours(demo_contours_img, [c], -1, (0, 255, 255), 2)  # Galben pentru contururi

        cv2.rectangle(demo_contours_img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Albastru pentru Global BBox
        cv2.putText(demo_contours_img, "Motion Global BBox", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                    2)
        cv2.imwrite("2e_motion_contours_and_bbox.jpg", demo_contours_img)
        print("Salvat: Cadrele 2b -> 2e pentru fazele de Motion Detection")

    # ==========================================
    # ETAPA 3: SMART BBOX (Decizia hibridă)
    # ==========================================
    print("Etapa 3: Decizia Smart BBox...")
    final_box = (0, 0, w_img, h_img)  # Default full frame

    # YOLO are prioritate, conform descrierii tale
    if bbox_yolo is not None:
        final_box = bbox_yolo
    elif bbox_motion is not None:
        final_box = bbox_motion

    x, y, w, h = final_box
    cx, cy = x + w // 2, y + h // 2

    # Verificăm dacă respectă cerința minimă de 140x140
    target_w, target_h = max(w, MIN_CROP_SIZE[0]), max(h, MIN_CROP_SIZE[1])

    fx1, fy1 = max(0, cx - target_w // 2), max(0, cy - target_h // 2)
    fx2, fy2 = min(w_img, fx1 + target_w), min(h_img, fy1 + target_h)

    # Corecție la marginea imaginii
    if fx2 == w_img: fx1 = max(0, fx2 - target_w)
    if fy2 == h_img: fy1 = max(0, fy2 - target_h)

    bbox_smart = (int(fx1), int(fy1), int(fx2 - fx1), int(fy2 - fy1))

    # Desenăm pe un cadru reprezentativ
    if found_person_yolo and yolo_demo_frame is not None:
        smart_demo_frame = frames[yolo_demo_idx].copy()
    else:
        smart_demo_frame = frames[motion_demo_idx if found_motion else 0].copy()

    cv2.rectangle(smart_demo_frame, (bbox_smart[0], bbox_smart[1]),
                  (bbox_smart[0] + bbox_smart[2], bbox_smart[1] + bbox_smart[3]), (0, 255, 0), 3)
    cv2.putText(smart_demo_frame, "Smart BBox Final", (bbox_smart[0], bbox_smart[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    cv2.imwrite("3_smart_bbox_decision.jpg", smart_demo_frame)
    print("Salvat: 3_smart_bbox_decision.jpg")

    # ==========================================
    # ETAPA 4: CROP FINAL + PADDING
    # ==========================================
    print("Etapa 4: Generarea cadrului final...")
    frame_for_crop = frames[yolo_demo_idx if found_person_yolo else (motion_demo_idx if found_motion else 0)]
    final_output = process_frame_final(frame_for_crop, bbox_smart, TARGET_SIZE)
    cv2.imwrite("4_final_cropped_padded.jpg", final_output)
    print("Salvat: 4_final_cropped_padded.jpg")

    print("====== TOATE ETAPELE AU FOST FINALIZATE CU SUCCES ======")


# ==========================================
# PUNCT DE START
# ==========================================
if __name__ == '__main__':
    # Schimbă cu căile tale valide de pe disc
    TEST_VIDEO = "../../Datasets/AI4RiSK/2/5eYWIed6quX6bSxi.mp4"
    YOLO_MODEL = "../training_ai4risk_x3dL_binary/yolov8m.pt"

    if Path(TEST_VIDEO).exists():
        save_detailed_preprocessing_steps(TEST_VIDEO, YOLO_MODEL)
    else:
        print(f"Modifică calea 'TEST_VIDEO'. Fișierul nu a fost găsit: {TEST_VIDEO}")