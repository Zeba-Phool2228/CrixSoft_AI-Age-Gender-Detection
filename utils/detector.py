"""
detector.py
-----------
Core AI logic for the Age & Gender Detection project.
Combines two models:
  1. OpenCV Face Detector -> finds WHERE faces are in an image
  2. ViT Age/Gender Model  -> predicts age & gender for a given face crop

Face detection uses a multi-scale ("tiled") strategy for large/crowded
images, with:
  1. IoU-based NMS to merge near-duplicate boxes
  2. A follow-up center-distance-based merge to catch same-face duplicates
     that IoU-NMS misses because the boxes differ too much in size/position
     (e.g. one from the full-image pass, one from a tile pass)
  3. Minimum-size and aspect-ratio filters to reject false positives
     (hands, objects)

Face crops sent to the age/gender model are SQUARE (proportional margin)
and lighting-normalized with CLAHE before prediction. Age is reported both
as a point estimate and as a realistic range.
"""

import os
import sys
import cv2
import torch
from PIL import Image

VIT_DIR = os.path.join("models", "age_gender_vit")
sys.path.insert(0, VIT_DIR)

from transformers import AutoConfig, AutoImageProcessor
from model import AgeGenderViTModel

FACE_PROTO = os.path.join("models", "face_detector", "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join("models", "face_detector", "opencv_face_detector_uint8.pb")

FACE_CONFIDENCE_THRESHOLD = 0.55
NMS_IOU_THRESHOLD = 0.30
TILING_MIN_DIMENSION = 700
TILE_GRID = (2, 2)
TILE_OVERLAP = 0.25

MIN_FACE_SIZE = 24
MIN_ASPECT_RATIO = 0.55
MAX_ASPECT_RATIO = 1.55

DUPLICATE_CENTER_DISTANCE_RATIO = 0.75

SQUARE_CROP_MARGIN = 0.35
AGE_ERROR_MARGIN = 5


class AgeGenderDetector:
    """Loads both models once, then exposes a single analyze() method."""

    def __init__(self):
        print("[detector] Loading face detector...")
        self.face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

        print("[detector] Loading age/gender ViT model (local files only)...")
        config = AutoConfig.from_pretrained(VIT_DIR, local_files_only=True)
        self.vit_model = AgeGenderViTModel.from_pretrained(
            VIT_DIR, config=config, local_files_only=True
        )
        self.vit_model.eval()
        self.processor = AutoImageProcessor.from_pretrained(
            VIT_DIR, local_files_only=True, do_center_crop=False
        )
        print("[detector] Ready.")

    def _detect_faces_raw(self, frame, conf_threshold):
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return []
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    results.append(((x1, y1, x2, y2), float(confidence)))
        return results

    def _detect_tiled(self, frame, conf_threshold):
        h, w = frame.shape[:2]
        rows, cols = TILE_GRID
        step_h = h // rows
        step_w = w // cols
        pad_h = int(step_h * TILE_OVERLAP)
        pad_w = int(step_w * TILE_OVERLAP)

        results = []
        for r in range(rows):
            for c in range(cols):
                y0 = max(0, r * step_h - pad_h)
                x0 = max(0, c * step_w - pad_w)
                y1 = min(h, (r + 1) * step_h + pad_h)
                x1 = min(w, (c + 1) * step_w + pad_w)
                tile = frame[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                for (tx1, ty1, tx2, ty2), conf in self._detect_faces_raw(tile, conf_threshold):
                    results.append(((tx1 + x0, ty1 + y0, tx2 + x0, ty2 + y0), conf))
        return results

    @staticmethod
    def _merge_close_duplicates(candidates, distance_ratio=DUPLICATE_CENTER_DISTANCE_RATIO):
        """
        Second-pass duplicate removal, run AFTER IoU-based NMS. Two
        detections are considered the same physical face if their centers
        are close relative to their own size — catches duplicates across
        the full-image pass and tile passes that plain IoU misses.
        """
        sorted_candidates = sorted(candidates, key=lambda c: c[1], reverse=True)
        kept = []
        for box, conf in sorted_candidates:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            size = max(x2 - x1, y2 - y1)

            is_duplicate = False
            for kbox, _ in kept:
                kx1, ky1, kx2, ky2 = kbox
                kcx, kcy = (kx1 + kx2) / 2.0, (ky1 + ky2) / 2.0
                ksize = max(kx2 - kx1, ky2 - ky1)
                avg_size = (size + ksize) / 2.0
                if avg_size <= 0:
                    continue
                dist = ((cx - kcx) ** 2 + (cy - kcy) ** 2) ** 0.5
                if dist < distance_ratio * avg_size:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append((box, conf))

        return kept

    def detect_faces(self, frame, conf_threshold=FACE_CONFIDENCE_THRESHOLD):
        h, w = frame.shape[:2]
        candidates = self._detect_faces_raw(frame, conf_threshold)

        if max(h, w) >= TILING_MIN_DIMENSION:
            candidates += self._detect_tiled(frame, conf_threshold)

        if not candidates:
            return []

        boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2), conf in candidates]
        confidences = [conf for box, conf in candidates]

        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, conf_threshold, NMS_IOU_THRESHOLD)
        if len(indices) == 0:
            return []

        survivors = []
        for i in indices.flatten():
            (x1, y1, x2, y2), conf = candidates[i]
            box_w, box_h = x2 - x1, y2 - y1

            if box_w < MIN_FACE_SIZE or box_h < MIN_FACE_SIZE:
                continue

            aspect = box_w / box_h if box_h else 0
            if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
                continue

            survivors.append(((x1, y1, x2, y2), conf))

        deduped = self._merge_close_duplicates(survivors)
        return [box for box, _ in deduped]

    @staticmethod
    def _get_square_crop_box(frame_shape, box, margin=SQUARE_CROP_MARGIN):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1
        cx, cy = x1 + box_w / 2.0, y1 + box_h / 2.0

        side = max(box_w, box_h) * (1.0 + margin)
        half = side / 2.0

        nx1 = int(round(cx - half))
        ny1 = int(round(cy - half))
        nx2 = int(round(cx + half))
        ny2 = int(round(cy + half))

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(w, nx2)
        ny2 = min(h, ny2)

        return nx1, ny1, nx2, ny2

    @staticmethod
    def _normalize_lighting(face_bgr):
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        normalized_lab = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    def predict_age_gender(self, pil_face_image):
        inputs = self.processor(images=pil_face_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        raw_age = logits[0, 0].item()
        age = int(round(raw_age))
        age = max(0, min(100, age))
        age_lo = max(0, age - AGE_ERROR_MARGIN)
        age_hi = min(100, age + AGE_ERROR_MARGIN)

        gender_prob_female = logits[0, 1].item()
        gender_prob_male = 1.0 - gender_prob_female
        gender = "Female" if gender_prob_female >= 0.5 else "Male"
        confidence = max(gender_prob_female, gender_prob_male)

        return age, (age_lo, age_hi), gender, confidence

    def analyze(self, frame, conf_threshold=FACE_CONFIDENCE_THRESHOLD):
        boxes = self.detect_faces(frame, conf_threshold)
        annotated = frame.copy()
        results = []

        for box in boxes:
            x1, y1, x2, y2 = box
            fx1, fy1, fx2, fy2 = self._get_square_crop_box(frame.shape, box)
            face_crop = frame[fy1:fy2, fx1:fx2]
            if face_crop.size == 0:
                continue

            face_crop = self._normalize_lighting(face_crop)
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)

            age, age_range, gender, gender_conf = self.predict_age_gender(pil_face)
            results.append({
                "box": (x1, y1, x2, y2),
                "age": age,
                "age_range": age_range,
                "gender": gender,
                "gender_confidence": gender_conf,
            })
            self._draw_result(annotated, x1, y1, x2, y2, age, gender, gender_conf)

        return annotated, results

    @staticmethod
    def _draw_result(frame, x1, y1, x2, y2, age, gender, gender_conf):
        color = (72, 219, 251) if gender == "Male" else (185, 103, 255)
        label = "{}, ~{}".format(gender, age)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "{:.0f}%".format(gender_conf * 100), (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)