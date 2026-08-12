import sys
import os

sys.path.insert(0, os.path.join("models", "age_gender_vit"))

import cv2
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor
from model import AgeGenderViTModel

FACE_PROTO = os.path.join("models", "face_detector", "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join("models", "face_detector", "opencv_face_detector_uint8.pb")
VIT_DIR = os.path.join("models", "age_gender_vit")
TEST_IMAGE = os.path.join("assets", "test_face.jpg")
PADDING = 20

print("Loading face detector...")
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

print("Loading age/gender ViT model (local only)...")
config = AutoConfig.from_pretrained(VIT_DIR, local_files_only=True)
vit_model = AgeGenderViTModel.from_pretrained(VIT_DIR, config=config, local_files_only=True)
vit_model.eval()
processor = AutoImageProcessor.from_pretrained(VIT_DIR, local_files_only=True, do_center_crop=False)


def predict_age_gender_for_crop(pil_face_image):
    inputs = processor(images=pil_face_image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    age = int(round(logits[0, 0].item()))
    age = max(0, min(100, age))
    gender_prob_female = logits[0, 1].item()
    gender_prob_male = 1.0 - gender_prob_female
    gender = "Female" if gender_prob_female >= 0.5 else "Male"
    confidence = max(gender_prob_female, gender_prob_male)
    return age, gender, confidence


print("Loading image:", TEST_IMAGE)
frame = cv2.imread(TEST_IMAGE)
h, w = frame.shape[:2]

blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
face_net.setInput(blob)
detections = face_net.forward()

faces_found = 0
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.65:
        faces_found += 1
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        fy1, fy2 = max(0, y1 - PADDING), min(h - 1, y2 + PADDING)
        fx1, fx2 = max(0, x1 - PADDING), min(w - 1, x2 + PADDING)
        face_crop = frame[fy1:fy2, fx1:fx2]

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb)

        age, gender, gender_conf = predict_age_gender_for_crop(pil_face)

        print("\nFace {}:".format(faces_found))
        print("  Box:", (x1, y1, x2, y2))
        print("  Age:", age)
        print("  Gender:", gender, "({:.1f}%)".format(gender_conf * 100))

        # draw on the image
        color = (72, 219, 251) if gender == "Male" else (185, 103, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "{}, {}".format(gender, age)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

print("\nTotal faces processed:", faces_found)

output_path = os.path.join("assets", "test_output.jpg")
cv2.imwrite(output_path, frame)
print("Annotated image saved to:", output_path)