import cv2
import os

FACE_PROTO = os.path.join(
    "models", "face_detector", "opencv_face_detector.pbtxt"
)

FACE_MODEL = os.path.join(
    "models", "face_detector", "opencv_face_detector_uint8.pb"
)

TEST_IMAGE = os.path.join(
    "assets", "test_face.jpg"
)

print("Loading face detector...")

face_net = cv2.dnn.readNet(
    FACE_MODEL,
    FACE_PROTO
)

print("Loading image:", TEST_IMAGE)

frame = cv2.imread(TEST_IMAGE)

if frame is None:
    raise FileNotFoundError(
        f"Could not load image: {TEST_IMAGE}"
    )

h, w = frame.shape[:2]

print("Image size:", w, "x", h)

print("Running face detection...")

blob = cv2.dnn.blobFromImage(
    frame,
    1.0,
    (300, 300),
    [104, 117, 123],
    swapRB=False,
    crop=False
)

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

        print(
            "Face {}: confidence={:.2f}, box=({},{},{},{})".format(
                faces_found,
                confidence,
                x1,
                y1,
                x2,
                y2
            )
        )

print("\nTotal faces detected:", faces_found)