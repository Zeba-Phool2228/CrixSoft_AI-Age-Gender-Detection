"""
test_model.py
-------------
Diagnostic test for the existing Age/Gender detection pipeline.

IMPORTANT:
- Does NOT modify detector.py
- Does NOT modify app.py
- Does NOT modify model weights
- Does NOT train anything
- Only measures current predictions
"""

import os
import sys
import cv2

# Make project root available
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.detector import AgeGenderDetector


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

TEST_IMAGES = [
    "face1.jpg",
    "face2.jpg",
    "face3.jpg",
    "group1.jpg",
    "group2.jpg",
    "group3.jpg",
]

CONFIDENCE_THRESHOLD = 0.55


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def print_separator():
    print("\n" + "=" * 80)


def test_image(detector, image_name):
    image_path = os.path.join(ASSETS_DIR, image_name)

    print_separator()
    print(f"IMAGE: {image_name}")
    print(f"PATH : {image_path}")

    if not os.path.exists(image_path):
        print("STATUS: FILE NOT FOUND")
        return

    frame = cv2.imread(image_path)

    if frame is None:
        print("STATUS: COULD NOT READ IMAGE")
        return

    height, width = frame.shape[:2]

    print(f"IMAGE SIZE: {width} x {height}")

    # -----------------------------------------------------
    # Face detection only
    # -----------------------------------------------------

    boxes = detector.detect_faces(
        frame,
        conf_threshold=CONFIDENCE_THRESHOLD
    )

    print(f"FACES DETECTED: {len(boxes)}")

    if not boxes:
        print("No faces detected.")
        return

    # -----------------------------------------------------
    # Detailed prediction for every detected face
    # -----------------------------------------------------

    for index, box in enumerate(boxes, start=1):

        x1, y1, x2, y2 = box

        face_width = x2 - x1
        face_height = y2 - y1

        print("\n" + "-" * 70)
        print(f"FACE #{index}")

        print(f"Bounding Box : ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"Face Size    : {face_width} x {face_height}")

        # Square crop using the SAME method as detector.py
        fx1, fy1, fx2, fy2 = detector._get_square_crop_box(
            frame.shape,
            box
        )

        print(
            f"Square Crop  : ({fx1}, {fy1}) -> ({fx2}, {fy2})"
        )

        face_crop = frame[fy1:fy2, fx1:fx2]

        if face_crop.size == 0:
            print("Prediction skipped: empty crop.")
            continue

        # SAME preprocessing as detector.py
        face_crop = detector._normalize_lighting(face_crop)

        face_rgb = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2RGB
        )

        from PIL import Image

        pil_face = Image.fromarray(face_rgb)

        # -------------------------------------------------
        # Model prediction
        # -------------------------------------------------

        age, age_range, gender, gender_conf = (
            detector.predict_age_gender(pil_face)
        )

        female_probability = None
        male_probability = None

        # Access raw model output so we can inspect
        # the actual gender probability.
        inputs = detector.processor(
            images=pil_face,
            return_tensors="pt"
        )

        import torch

        with torch.no_grad():
            outputs = detector.vit_model(**inputs)

        logits = (
            outputs.logits
            if hasattr(outputs, "logits")
            else outputs[0]
        )

        raw_age = float(logits[0, 0].item())
        female_probability = float(logits[0, 1].item())
        male_probability = 1.0 - female_probability

        print("\nPREDICTION")
        print(f"Age              : {age}")
        print(f"Age Range        : {age_range[0]} - {age_range[1]}")
        print(f"Raw Age Output   : {raw_age:.4f}")

        print(f"Gender           : {gender}")
        print(
            f"Female Probability: "
            f"{female_probability * 100:.2f}%"
        )
        print(
            f"Male Probability  : "
            f"{male_probability * 100:.2f}%"
        )
        print(
            f"Gender Confidence : "
            f"{gender_conf * 100:.2f}%"
        )

        # -------------------------------------------------
        # Confidence interpretation
        # -------------------------------------------------

        if gender_conf >= 0.80:
            confidence_level = "HIGH"
        elif gender_conf >= 0.65:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        print(f"Confidence Level  : {confidence_level}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print_separator()
    print("AI AGE & GENDER MODEL DIAGNOSTIC TEST")
    print_separator()

    print("Loading existing detector...")
    detector = AgeGenderDetector()

    print("Detector loaded successfully.")

    for image_name in TEST_IMAGES:
        test_image(detector, image_name)

    print_separator()
    print("DIAGNOSTIC TEST COMPLETE")
    print_separator()


if __name__ == "__main__":
    main()