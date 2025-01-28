import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms
from .face_recognition.arcface.model import iresnet_inference
from .face_recognition.arcface.utils import compare_encodings
from .face_alignment.alignment import norm_crop
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

feature_path = os.path.join(BASE_DIR, "datasets", "face_features", "feature")

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

def get_face_embedding(face_image, landmarks):
    """Extract features for a given face image."""
    # Align the face using the landmarks
    aligned_face = norm_crop(face_image, landmarks)

    # Convert the aligned face to RGB
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

    # Preprocess the aligned face
    face_tensor = face_preprocess(aligned_face).unsqueeze(0).to(device)

    # Generate embedding
    emb = recognizer(face_tensor).cpu().detach().numpy()
    return emb / np.linalg.norm(emb)

def face_rec(face_image, landmarks):
    """Match face image against known embeddings."""
    query_emb = get_face_embedding(face_image, landmarks)
    score, id_min = compare_encodings(query_emb, images_embs)
    return score, images_names[id_min] if score > 0.5 else None

def is_face_in_person_box(face_box, person_box, iou_threshold=0.5):
    """Check if a face bounding box is within a model bounding box."""
    x1 = max(face_box[0], person_box[0])
    y1 = max(face_box[1], person_box[1])
    x2 = min(face_box[2], person_box[2])
    y2 = min(face_box[3], person_box[3])

    if x2 <= x1 or y2 <= y1:
        return False

    face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
    intersection_area = (x2 - x1) * (y2 - y1)

    return intersection_area / face_area > iou_threshold

def recognize_faces_in_persons(frame, person_bboxes, track_ids):
    current_time = time.time()
    print("\n--- Starting face recognition process ---")

    # YOLO face detection
    start_time = time.time()
    face_results = yolo_model.predict(frame, conf=0.7)
    face_boxes = [list(map(int, bbox)) for result in face_results for bbox in result.boxes.xyxy]
    print(f"YOLO face detection completed in {time.time() - start_time:.2f} seconds")
    print(f"Number of faces detected: {len(face_boxes)}")

    # Loop through detected faces
    print(np.array(face_boxes).tolist(), track_ids)
    for i, (face_box, track_id) in enumerate(zip(np.array(face_boxes).tolist(), track_ids)):
        print(f"\nProcessing face {i + 1}/{len(face_boxes)}")
        x1, y1, x2, y2 = face_box
        cropped_face = frame[y1:y2, x1:x2]

        # Detect landmarks
        start_time = time.time()
        from mtcnn import MTCNN
        landmark_detector = MTCNN()
        landmarks = landmark_detector.detect_faces(cropped_face)
        print(f"Landmark detection completed in {time.time() - start_time:.2f} seconds")
        if not landmarks:
            print("No landmarks detected for this face.")
            continue
        keypoints = landmarks[0]["keypoints"]
        face_landmarks = np.array([
            keypoints["left_eye"],
            keypoints["right_eye"],
            keypoints["nose"],
            keypoints["mouth_left"],
            keypoints["mouth_right"],
        ])

        # Perform face recognition
        start_time = time.time()
        score, name = face_rec(cropped_face, face_landmarks)
        print(f"Face recognition completed in {time.time() - start_time:.2f} seconds")
        print(f"Recognition result: {'SIETIAN ' + name if name else 'UNKNOWN'}")
        detection = f"SIETIAN {name}" if name else "UNKNOWN"


    start_time = time.time()
    states = ["UNDETERMINED"] * len(person_bboxes)
    print(f"Linking face data to track_ids completed in {time.time() - start_time:.2f} seconds")

    print("--- Face recognition process completed ---\n")
    return frame, states

