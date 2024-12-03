
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms
from .FaceTraacker_test import FaceTracker
from .face_recognition.arcface.model import iresnet_inference
from .face_recognition.arcface.utils import compare_encodings, read_features
import os
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO model initialization
yolo_model = YOLO(os.path.join(BASE_DIR,"face_detection","yolo","weights","yolo11n-face.pt"))

# ArcFace model initialization
recognizer = iresnet_inference(
    model_name="r100",
    path=os.path.join(
        BASE_DIR, "face_recognition", "arcface", "weights", "glink360k_cosface_r100_fp16_0.1.pth"
    ),
    device=device,
)

# # Preloaded face embeddings and names
# images_names, images_embs = read_features(
#     feature_path=os.path.join(
#         BASE_DIR, "datasets", "face_features", "arcface100_featureALL"
#     )
# )

feature_path = os.path.join(BASE_DIR, "datasets", "face_features", "glink360k_featuresALL")

# Construct paths for the two .npy files
images_name_path = os.path.join(feature_path, "images_name.npy")
images_emb_path = os.path.join(feature_path, "images_emb.npy")
# Load the .npy files
images_names = np.load(images_name_path)
images_embs = np.load(images_emb_path)


# Preprocessing for ArcFace input
face_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def get_face_embedding(face_image):
    """Extract features for a given face image."""
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_tensor = face_preprocess(face_image).unsqueeze(0).to(device)
    emb = recognizer(face_tensor).cpu().detach().numpy()
    return emb / np.linalg.norm(emb)


def face_rec(face_image):
    """Match face image against known embeddings."""
    query_emb = get_face_embedding(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    return score, images_names[id_min] if score > 0.5 else None


def is_face_in_person_box(face_box, person_box, iou_threshold=0.5):
    """Check if a face bounding box is within a person bounding box."""
    x1 = max(face_box[0], person_box[0])
    y1 = max(face_box[1], person_box[1])
    x2 = min(face_box[2], person_box[2])
    y2 = min(face_box[3], person_box[3])

    if x2 <= x1 or y2 <= y1:
        return False

    face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
    intersection_area = (x2 - x1) * (y2 - y1)

    return intersection_area / face_area > iou_threshold


def recognize_faces_in_persons(frame, person_bboxes, face_tracker: FaceTracker):
    current_time = time.time()

    # YOLO face detection
    face_results = yolo_model.predict(frame, conf=0.7)
    face_boxes = [list(map(int, bbox)) for result in face_results for bbox in result.boxes.xyxy]

    # Update each detected face
    for face_box in face_boxes:
        x1, y1, x2, y2 = face_box
        cropped_face = frame[y1:y2, x1:x2]
        score, name = face_rec(cropped_face)
        detection = f"SIETIAN {name}" if name else "UNKNOWN"
        face_tracker.update_face(face_box, detection, current_time)

    # Update face states periodically
    face_tracker.update_states(current_time)

    # Draw bounding boxes and labels
    for data in face_tracker.get_tracked_faces():
        x1, y1, x2, y2 = data["box"]
        if (data["state"] == 'UNKNOWN'):
            color = (0, 0, 255)
        if (data["state"] == 'PENDING'):
            color = (255, 0, 0)
        if data["state"].startswith("SIETIAN"):
            color = (0 , 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, data["state"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Check person boxes
    states = ["UNDETERMINED"] * len(person_bboxes)
    for idx, person_box in enumerate(person_bboxes):
        for data in face_tracker.get_tracked_faces():
            if is_face_in_person_box(data["box"], person_box):
                states[idx] = data["state"]
                break

    return frame, states