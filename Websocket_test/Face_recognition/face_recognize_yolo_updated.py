# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO
# from torchvision import transforms
# from .face_recognition.arcface.model import iresnet_inference
# from .face_recognition.arcface.utils import compare_encodings, read_features
# import os
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#
# from collections import defaultdict
# import time
#
# # Track face states
# face_tracker = defaultdict(lambda: {"state": "UNDETERMINED", "last_seen": time.time(), "box": None})
#
#
# timeout = 2  # seconds before marking as UNKNOWN
# lost_threshold = 5  # seconds before removing from the tracker
#
#
# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # YOLO model initialization
# yolo_model = YOLO(os.path.join(BASE_DIR,"face_detection","yolo","weights","yolo11n-face.pt"))
#
# # ArcFace model initialization
#
# recognizer = iresnet_inference(
#     model_name="r100",
#     path=os.path.join(
#         BASE_DIR, "face_recognition", "arcface", "weights", "arcface_r100.pth"
#     ),
#     device=device,
# )
# # Preloaded face embeddings and names
# images_names, images_embs = read_features(
#     feature_path=os.path.join(
#         BASE_DIR, "datasets", "face_features", "featureTEST"
#     )
# )
# # Preprocessing for ArcFace input
# face_preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((112, 112)),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])
#
#
# def update_current_frame(frame):
#     global current_frame
#     current_frame += 1
#     return frame
#
# def get_face_embedding(face_image):
#     """Extract features for a given face image."""
#     face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#     face_tensor = face_preprocess(face_image).unsqueeze(0).to(device)
#     emb = recognizer(face_tensor).cpu().detach().numpy()
#     return emb / np.linalg.norm(emb)
#
#
# def face_rec(face_image):
#     """Match face image against known embeddings."""
#     query_emb = get_face_embedding(face_image)
#     score, id_min = compare_encodings(query_emb, images_embs)
#     return score, images_names[id_min] if score > 0.5 else None
#
#
# def is_face_in_person_box(face_box, person_box, iou_threshold=0.5):
#     """Check if a face bounding box is within a person bounding box."""
#     x1 = max(face_box[0], person_box[0])
#     y1 = max(face_box[1], person_box[1])
#     x2 = min(face_box[2], person_box[2])
#     y2 = min(face_box[3], person_box[3])
#
#     if x2 <= x1 or y2 <= y1:
#         return False
#
#     face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
#     intersection_area = (x2 - x1) * (y2 - y1)
#
#     return intersection_area / face_area > iou_threshold
#
# def recognize_faces_in_persons(frame, person_bboxes):
#     """
#     Detect faces within person bounding boxes and classify them as SIETIAN, UNKNOWN, or UNDETERMINED.
#     Args:
#         frame: The video frame to process.
#         person_bboxes: List of person bounding boxes in the format [(x1, y1), (x2, y2)].
#     Returns:
#         modified_frame: Frame with person bounding boxes and recognition results.
#         states: List of states for each person box - SIETIAN, UNKNOWN, or UNDETERMINED.
#     """
#     global face_tracker
#     # Detect faces using YOLO
#     face_results = yolo_model.predict(frame, conf=0.5)
#     face_boxes = [
#         list(map(int, bbox)) for result in face_results for bbox in result.boxes.xyxy
#     ]
#
#     # Initialize states for person bounding boxes
#     states = ["UNDETERMINED"] * len(person_bboxes)
#
#     # Process each detected face
#     current_time = time.time()
#     updated_faces = set()
#
#     for face_box in face_boxes:
#         # Generate a temporary ID for the face using the bounding box
#         face_id = tuple(face_box)  # Simple ID based on coordinates
#         updated_faces.add(face_id)
#
#         # Crop and preprocess the face image
#         x1, y1, x2, y2 = face_box
#         cropped_face = frame[y1:y2, x1:x2]
#         score, name = face_rec(cropped_face)
#
#         state = f"SIETIAN ({name})" if name else "UNKNOWN"
#
#         # Draw bounding box and state
#         color = (0, 255, 0) if "SIETIAN" in state else (0, 0, 255)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(
#             frame,
#             state,
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             color,
#             2,
#         )
#
#
#     # Check person bounding boxes for state updates
#     for idx, person_box in enumerate(person_bboxes):
#         for face_id, face_data in face_tracker.items():
#             if is_face_in_person_box(face_data["box"], person_box):
#                 states[idx] = face_data["state"]
#                 break
#
#     return frame, states
#
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms
from .face_recognition.arcface.model import iresnet_inference
from .face_recognition.arcface.utils import compare_encodings, read_features
import os
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))



timeout = 2  # seconds before marking as UNKNOWN
lost_threshold = 5  # seconds before removing from the tracker


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO model initialization
yolo_model = YOLO(os.path.join(BASE_DIR,"face_detection","yolo","weights","yolo11n-face.pt"))

# ArcFace model initialization

recognizer = iresnet_inference(
    model_name="r100",
    path=os.path.join(
        BASE_DIR, "face_recognition", "arcface", "weights", "arcface_r100.pth"
    ),
    device=device,
)
# Preloaded face embeddings and names
images_names, images_embs = read_features(
    feature_path=os.path.join(
        BASE_DIR, "datasets", "face_features", "featureTEST"
    )
)
# Preprocessing for ArcFace input
face_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def update_current_frame(frame):
    global current_frame
    current_frame += 1
    return frame

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

from .sort import Sort # Import SORT tracker

# Initialize SORT tracker
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

def recognize_faces_in_persons(frame, person_bboxes):
    """
    Detect faces within person bounding boxes and classify them as SIETIAN, UNKNOWN, or UNDETERMINED.
    Args:
        frame: The video frame to process.
        person_bboxes: List of person bounding boxes in the format [(x1, y1), (x2, y2)].
    Returns:
        modified_frame: Frame with person bounding boxes and recognition results.
        states: List of states for each person box - SIETIAN, UNKNOWN, or UNDETERMINED.
    """
    global face_tracker

    # Detect faces using YOLO
    face_results = yolo_model.predict(frame, conf=0.5)
    face_boxes = [
        list(map(int, bbox)) for result in face_results for bbox in result.boxes.xyxy
    ]

    # Update SORT tracker with detected face boxes
    tracked_faces = tracker.update(np.array(face_boxes))

    # Initialize states for person bounding boxes
    states = ["UNDETERMINED"] * len(person_bboxes)

    # Process tracked faces
    for face_data in tracked_faces:
        # Ensure that face_data contains the bounding box and, if available, the track ID
        if len(face_data) == 4:
            x1, y1, x2, y2 = map(int, face_data)  # Only extract bbox if no track_id is present
            track_id = -1  # Optional: Set a default value if track_id is missing
        elif len(face_data) == 5:
            x1, y1, x2, y2, track_id = map(int, face_data)  # Extract bbox and track_id
        else:
            print(f"Unexpected face_data format: {face_data}")
            continue  # Skip if the format is unexpected

        cropped_face = frame[y1:y2, x1:x2]
        score, name = face_rec(cropped_face)

        state = f"SIETIAN ({name})" if name else "UNKNOWN"

        # Update face tracker for visualization
        face_tracker[track_id] = {
            "box": [x1, y1, x2, y2],
            "state": state,
        }

        # Draw bounding box and state
        color = (0, 255, 0) if "SIETIAN" in state else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {track_id}: {state}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    # Check person bounding boxes for state updates
    for idx, person_box in enumerate(person_bboxes):
        for track_id, face_data in face_tracker.items():
            if is_face_in_person_box(face_data["box"], person_box):
                states[idx] = face_data["state"]
                break

    return frame, states
