import torch
import cv2
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from ultralytics import YOLO
import sys
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from boxmot import DeepOCSORT

sys.path.append('/mnt/project')

from student_model import R3D18Student
from teacher_model import R3D18Violence
from config import R3DTransferConfig


class VideoInference:
    def __init__(self, model_path_student, model_path_teacher, config, yolo_model='yolo11m-pose.pt',
                 overlap_threshold=0.3, reid_model_path='osnet_x0_25_msmt17.pt',
                 skeleton_alpha=0.35, skeleton_line_thickness=2, skeleton_point_radius=3):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

        self.skeleton_alpha = skeleton_alpha
        self.skeleton_line_thickness = skeleton_line_thickness
        self.skeleton_point_radius = skeleton_point_radius

        self.student_model = R3D18Student(num_classes=2, pretrained=False).to(self.device)
        checkpoint_student = torch.load(model_path_student, map_location=self.device)
        self.student_model.load_state_dict(checkpoint_student['model_state_dict'])
        self.student_model.eval()

        self.teacher_model = R3D18Violence(num_classes=2, pretrained=False).to(self.device)
        checkpoint_teacher = torch.load(model_path_teacher, map_location=self.device)
        self.teacher_model.load_state_dict(checkpoint_teacher['model_state_dict'])
        self.teacher_model.eval()

        self.yolo = YOLO(yolo_model)

        self.tracker = DeepOCSORT(
            model_weights=Path(reid_model_path),
            device=self.device,
            fp16=True,
            max_age=60,
            det_thresh=0.25,
            iou_thresh=0.3
        )

        self.mean = torch.tensor(config.KINETICS_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(config.KINETICS_STD).view(3, 1, 1, 1)

        self.frame_buffer = deque(maxlen=16)
        self.current_prediction_student = None
        self.current_confidence_student = 0.0
        self.current_prediction_teacher = None
        self.current_confidence_teacher = 0.0
        self.current_smart_crop_bbox = None

        self.overlap_threshold = overlap_threshold

        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        self.prev_positions = defaultdict(lambda: deque(maxlen=30))
        self.crowd_entropy = 0.0
        self.local_density = 0.0

    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (112, 112))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        return frame_normalized

    def get_smart_crop_bbox(self, heatmap, original_h, original_w, threshold=0.6):
        heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
        binary_mask = (heatmap_resized > threshold).astype(np.uint8) * 255

        if binary_mask.sum() == 0:
            return None

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        new_w = max(w, 30)
        new_h = max(h, 30)

        center_x = x + w // 2
        center_y = y + h // 2

        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)

        if new_x + new_w > original_w:
            new_x = original_w - new_w
        if new_y + new_h > original_h:
            new_y = original_h - new_h

        return (max(0, int(new_x)), max(0, int(new_y)), int(new_w), int(new_h))

    def calculate_overlap(self, bbox1, bbox2):
        x1_1, y1_1, w1, h1 = bbox1
        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1

        x1_2, y1_2, x2_2, y2_2 = bbox2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        if bbox2_area == 0:
            return 0.0

        overlap_ratio = intersection_area / bbox2_area
        return overlap_ratio

    def calculate_crowd_metrics(self, current_positions):
        if len(current_positions) < 2:
            return 0.0, 0.0

        velocities = []
        centers = []

        for track_id, positions in current_positions.items():
            if len(positions) >= 2:
                pos_array = np.array(positions)
                velocity = pos_array[-1] - pos_array[-2]
                velocities.append(velocity)
                centers.append(pos_array[-1])

        if len(velocities) < 2:
            return 0.0, 0.0

        velocities = np.array(velocities)
        centers = np.array(centers)

        speeds = np.linalg.norm(velocities, axis=1)
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])

        speed_bins = 10
        angle_bins = 8

        speed_hist, _ = np.histogram(speeds, bins=speed_bins, density=True)
        angle_hist, _ = np.histogram(angles, bins=angle_bins, density=True)

        speed_hist = speed_hist[speed_hist > 0]
        angle_hist = angle_hist[angle_hist > 0]

        entropy_speed = entropy(speed_hist) if len(speed_hist) > 0 else 0.0
        entropy_angle = entropy(angle_hist) if len(angle_hist) > 0 else 0.0

        crowd_entropy = (entropy_speed + entropy_angle) / 2.0

        distances = pdist(centers)
        if len(distances) > 0:
            avg_distance = np.mean(distances)
            local_density = 1.0 / (avg_distance + 1e-6)
        else:
            local_density = 0.0

        return crowd_entropy, local_density

    def predict_sequence(self):
        if len(self.frame_buffer) != 16:
            return None, 0.0, None, 0.0, None

        frames = list(self.frame_buffer)
        sequence = np.stack(frames, axis=0)
        sequence_tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2)
        sequence_tensor = (sequence_tensor - self.mean) / self.std
        input_tensor = sequence_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_student = self.student_model(input_tensor)
            probs_student = torch.softmax(output_student, dim=1)[0]
            pred_student = output_student.argmax(dim=1).item()
            conf_student = probs_student[pred_student].item()

        input_tensor_teacher = input_tensor.clone()
        input_tensor_teacher.requires_grad = True

        output_teacher = self.teacher_model(input_tensor_teacher, return_cam=True)
        probs_teacher = torch.softmax(output_teacher, dim=1)[0]
        pred_teacher = output_teacher.argmax(dim=1).item()
        conf_teacher = probs_teacher[pred_teacher].item()

        self.teacher_model.zero_grad()
        output_teacher[0, pred_teacher].backward()

        heatmap = self.teacher_model.get_spatial_cam_plus_plus(pred_teacher)
        heatmap_np = heatmap[0].detach().cpu().numpy() if heatmap is not None else None

        return pred_student, conf_student, pred_teacher, conf_teacher, heatmap_np

    def draw_predictions(self, frame, pred_student, conf_student, pred_teacher, conf_teacher):
        label_student = "VIOLENT" if pred_student == 1 else "NON-VIOLENT"
        color_student = (0, 0, 255) if pred_student == 1 else (0, 255, 0)

        label_teacher = "VIOLENT" if pred_teacher == 1 else "NON-VIOLENT"
        color_teacher = (0, 0, 255) if pred_teacher == 1 else (0, 255, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        text_student = f"Student: {label_student} {conf_student * 100:.1f}%"
        text_teacher = f"Teacher: {label_teacher} {conf_teacher * 100:.1f}%"

        cv2.rectangle(frame, (5, 5), (280, 50), (0, 0, 0), -1)
        cv2.putText(frame, text_student, (10, 20), font, font_scale, color_student, thickness)
        cv2.putText(frame, text_teacher, (10, 40), font, font_scale, color_teacher, thickness)

    def draw_metrics(self, frame):
        h, w = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        text1 = f"Crowd Entropy: {self.crowd_entropy:.3f}"
        text2 = f"Local Density: {self.local_density:.3f}"

        text_size1 = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]

        max_width = max(text_size1[0], text_size2[0])

        bg_x = w - max_width - 20
        bg_y = 5
        bg_w = max_width + 10
        bg_h = 50

        cv2.rectangle(frame, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 0), -1)

        entropy_color = (0, 255, 0) if self.crowd_entropy < 1.5 else (0, 165, 255) if self.crowd_entropy < 2.5 else (0,
                                                                                                                     0,
                                                                                                                     255)
        density_color = (0, 255, 0) if self.local_density < 0.01 else (0, 165, 255) if self.local_density < 0.02 else (
            0, 0, 255)

        cv2.putText(frame, text1, (bg_x + 5, bg_y + 20), font, font_scale, entropy_color, thickness)
        cv2.putText(frame, text2, (bg_x + 5, bg_y + 40), font, font_scale, density_color, thickness)

    def draw_smart_crop(self, frame, bbox, heatmap):
        if bbox is None:
            return

        x, y, w, h = bbox

        frame_h, frame_w = frame.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (frame_w, frame_h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        mask[y:y + h, x:x + w] = 255

        overlay = frame.copy()
        overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
        frame[mask > 0] = overlay[mask > 0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(frame, "Smart Crop", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def draw_tracking(self, frame, tracks, original_detections, original_keypoints, smart_crop_bbox):
        current_positions = {}
        overlay = frame.copy()

        if len(tracks) > 0:
            for trk in tracks:
                x1, y1, x2, y2 = map(int, trk[:4])
                track_id = int(trk[4])

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                self.prev_positions[track_id].append(np.array([center_x, center_y]))
                current_positions[track_id] = list(self.prev_positions[track_id])

                is_suspect = False
                if smart_crop_bbox is not None:
                    overlap = self.calculate_overlap(smart_crop_bbox, (x1, y1, x2, y2))
                    is_suspect = overlap >= self.overlap_threshold

                color = (0, 0, 255) if is_suspect else (0, 255, 0)

                if original_detections is not None and original_keypoints is not None:
                    max_iou = 0
                    best_kpt_idx = -1

                    track_box = (x1, y1, x2 - x1, y2 - y1)

                    for i, det in enumerate(original_detections):
                        dx1, dy1, dx2, dy2 = map(int, det[:4])
                        det_box = (dx1, dy1, dx2, dy2)

                        iou = self.calculate_overlap(track_box, det_box)
                        if iou > max_iou:
                            max_iou = iou
                            best_kpt_idx = i

                    if best_kpt_idx != -1 and max_iou > 0.3:
                        kpts = original_keypoints[best_kpt_idx].xy.cpu().numpy()[0]

                        for connection in self.skeleton_connections:
                            pt1_idx, pt2_idx = connection
                            if pt1_idx < len(kpts) and pt2_idx < len(kpts):
                                x1_kpt, y1_kpt = int(kpts[pt1_idx][0]), int(kpts[pt1_idx][1])
                                x2_kpt, y2_kpt = int(kpts[pt2_idx][0]), int(kpts[pt2_idx][1])
                                if x1_kpt > 0 and y1_kpt > 0 and x2_kpt > 0 and y2_kpt > 0:
                                    cv2.line(overlay, (x1_kpt, y1_kpt), (x2_kpt, y2_kpt), color,
                                             self.skeleton_line_thickness)

                        for kpt in kpts:
                            x_kpt, y_kpt = int(kpt[0]), int(kpt[1])
                            if x_kpt > 0 and y_kpt > 0:
                                cv2.circle(overlay, (x_kpt, y_kpt), self.skeleton_point_radius, color, -1)

            cv2.addWeighted(overlay, self.skeleton_alpha, frame, 1.0 - self.skeleton_alpha, 0, frame)

            for trk in tracks:
                x1, y1, x2, y2 = map(int, trk[:4])
                track_id = int(trk[4])

                is_suspect = False
                if smart_crop_bbox is not None:
                    overlap = self.calculate_overlap(smart_crop_bbox, (x1, y1, x2, y2))
                    is_suspect = overlap >= self.overlap_threshold

                color = (0, 0, 255) if is_suspect else (0, 255, 0)

                label = f"ID: {track_id}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                text_x = int(x1)
                text_y = int(y1 - 5)

                cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 4),
                              (text_x + text_size[0] + 4, text_y + 4), color, -1)
                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        self.crowd_entropy, self.local_density = self.calculate_crowd_metrics(current_positions)

    def process_video(self, video_path, slowdown_factor=3, display_scale=1.5):
        cap = cv2.VideoCapture(str(video_path))

        print(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int((1000 / fps) * slowdown_factor)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            preprocessed_frame = self.preprocess_frame(frame)
            self.frame_buffer.append(preprocessed_frame)

            if len(self.frame_buffer) == 16:
                pred_student, conf_student, pred_teacher, conf_teacher, heatmap = self.predict_sequence()
                if pred_student is not None:
                    self.current_prediction_student = pred_student
                    self.current_confidence_student = conf_student
                    self.current_prediction_teacher = pred_teacher
                    self.current_confidence_teacher = conf_teacher

                    if heatmap is not None:
                        h, w = frame.shape[:2]
                        self.current_smart_crop_bbox = self.get_smart_crop_bbox(
                            heatmap, h, w, threshold=self.config.SMART_CROP_THRESHOLD
                        )
            else:
                heatmap = None

            yolo_results = self.yolo.predict(frame, verbose=False, conf=0.25)[0]
            dets = yolo_results.boxes.data.cpu().numpy()

            if len(dets) > 0:
                tracks = self.tracker.update(dets, frame)
            else:
                tracks = np.empty((0, 8))

            display_frame = frame.copy()

            if self.current_smart_crop_bbox is not None and heatmap is not None:
                self.draw_smart_crop(display_frame, self.current_smart_crop_bbox, heatmap)

            original_detections = dets if len(dets) > 0 else None
            original_keypoints = yolo_results.keypoints if yolo_results.keypoints else None

            self.draw_tracking(display_frame, tracks, original_detections, original_keypoints,
                               self.current_smart_crop_bbox)

            if self.current_prediction_student is not None:
                self.draw_predictions(
                    display_frame,
                    self.current_prediction_student,
                    self.current_confidence_student,
                    self.current_prediction_teacher,
                    self.current_confidence_teacher
                )

            self.draw_metrics(display_frame)

            if display_scale != 1.0:
                new_width = int(display_frame.shape[1] * display_scale)
                new_height = int(display_frame.shape[0] * display_scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))

            cv2.imshow('Violence Detection', display_frame)

            frame_count += 1

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = R3DTransferConfig(dataset_name='Mix', use_smart_crop=True)

    student_model_path = config.get_student_model_path()
    teacher_model_path = config.get_heatmap_model_path(config.DATASET_NAME)

    if not student_model_path.exists():
        raise FileNotFoundError(f"Student model not found at {student_model_path}")

    if not teacher_model_path.exists():
        raise FileNotFoundError(f"Teacher model not found at {teacher_model_path}")

    video_path = Path("../../Datasets/Hockey/Violence/fi143_xvid.avi")
    reid_path = Path("osnet_x0_25_msmt17.pt")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")

    if not reid_path.exists():
        print(f"Warning: ReID model not found at {reid_path}.")

    inference = VideoInference(
        student_model_path,
        teacher_model_path,
        config,
        yolo_model='yolo11m-pose.pt',
        overlap_threshold=0.3,
        reid_model_path=str(reid_path),
        skeleton_alpha=0.35,
        skeleton_line_thickness=2,
        skeleton_point_radius=3
    )

    inference.process_video(video_path, slowdown_factor=20, display_scale=1.5)