import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from ultralytics import YOLO
from .face_recognition.arcface.model import iresnet_inference
from .face_recognition.arcface.utils import compare_encodings

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize YOLO face detection model
yolo_model = YOLO(os.path.join(BASE_DIR, "face_detection", "yolo", "weights", "yolo11n-face.pt"))

# Initialize ArcFace recognition model
recognizer = iresnet_inference(
    model_name="r100",
    path=os.path.join(
        BASE_DIR, "face_recognition", "arcface", "weights", "glink360k_cosface_r100_fp16_0.1.pth"
    ),
    device=device,
)

# Load pre-saved face features
feature_path = os.path.join(BASE_DIR, "datasets", "face_features", "glink360k_featuresALL")
images_name_path = os.path.join(feature_path, "images_name.npy")
images_emb_path = os.path.join(feature_path, "images_emb.npy")
images_names = np.load(images_name_path)
images_embs = np.load(images_emb_path)


@torch.no_grad()
def get_feature(face_image):
    """Extract features from a face image."""
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    emb_img_face = recognizer(face_image).cpu().numpy()
    return emb_img_face / np.linalg.norm(emb_img_face)


def recognize_face(face_image):
    """Recognize a face image."""
    query_emb = get_feature(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    return score[0], name


def process_faces(frame):
    face_results = yolo_model.predict(frame, conf=0.7)
    labels = []  # To store labels
    bboxes = []  # To store bounding boxes

    for result in face_results:
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cropped_face = frame[y1:y2, x1:x2]

            try:
                score, name = recognize_face(cropped_face)

                if score >= 0.5:
                    label = f"{name} ({score:.2f})"
                    color = (255, 0, 255)  # Magenta for recognized
                else:
                    label = "UNKNOWN"
                    color = (0, 165, 255)  # Orange for unknown

                # Add label and bbox to the lists
                labels.append((label, (x1, y1)))
                bboxes.append((x1, y1, x2, y2))

            except Exception as e:
                print(f"Error recognizing face: {e}")
                continue

    return labels, bboxes
#
#
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#
#     labels, bboxes = process_faces(frame)
#
#     # Draw bounding boxes and labels
#     for (label, (x1, y1)), (x1, y1, x2, y2) in zip(labels, bboxes):
#         # Draw bounding box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#         # Draw label
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     # Display the frame
#     cv2.imshow("Face Recognition", frame)
#
#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
