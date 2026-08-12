import sys
import os

sys.path.insert(0, os.path.join("models", "age_gender_vit"))

import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor
from model import AgeGenderViTModel

MODEL_DIR = os.path.join("models", "age_gender_vit")
TEST_IMAGE = os.path.join("assets", "test_face.jpg")

print("Loading model (local files only, no download)...")
config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
model = AgeGenderViTModel.from_pretrained(MODEL_DIR, config=config, local_files_only=True)
model.eval()

print("Loading image processor (center-crop disabled to avoid a config bug)...")
processor = AutoImageProcessor.from_pretrained(
    MODEL_DIR, local_files_only=True, do_center_crop=False
)

print("Loading image:", TEST_IMAGE)
image = Image.open(TEST_IMAGE).convert("RGB")

print("Preprocessing...")
inputs = processor(images=image, return_tensors="pt")

print("Running inference...")
with torch.no_grad():
    outputs = model(**inputs)

if hasattr(outputs, "logits"):
    logits = outputs.logits
elif isinstance(outputs, (tuple, list)):
    logits = outputs[0]
else:
    logits = outputs

age = int(round(logits[0, 0].item()))
age = max(0, min(100, age))
gender_prob_female = logits[0, 1].item()
gender_prob_male = 1.0 - gender_prob_female
gender = "Female" if gender_prob_female >= 0.5 else "Male"
gender_confidence = max(gender_prob_female, gender_prob_male)

print("\n--- RESULT ---")
print("Age:", age)
print("Gender:", gender)
print("Gender confidence: {:.1f}%".format(gender_confidence * 100))