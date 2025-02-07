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
feature_path = os.path.join(BASE_DIR, "datasets", "face_features", "feature")
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
    best_label = None  # Store the best match
    best_bbox = None
    best_score = 0.0

    for result in face_results:
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cropped_face = frame[y1:y2, x1:x2]

            try:
                score, name = recognize_face(cropped_face)

                if score >= 0.5 and score > best_score:
                    best_label = name
                    best_bbox = (x1, y1, x2, y2)
                    best_score = score

            except Exception as e:
                print(f"Error recognizing face: {e}")
                continue

    # Return the best match or "UNKNOWN" if no face was detected confidently
    if best_label:
        return best_label, [best_bbox]
    else:
        return "UNKNOWN", []
# def align_face(face_image):
#     """Simple face alignment using center crop"""
#     h, w = face_image.shape[:2]
#     size = min(h, w)
#     y_start = (h - size) // 2
#     x_start = (w - size) // 2
#     aligned = face_image[y_start:y_start + size, x_start:x_start + size]
#     return cv2.resize(aligned, (112, 112))
#
#
# def enhance_face_quality(face_image):
#     """Improve face image quality for better recognition"""
#     lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
#     l_channel, a_channel, b_channel = cv2.split(lab)
#
#     # CLAHE for illumination normalization
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l_enhanced = clahe.apply(l_channel)
#
#     # Edge-preserving denoising
#     denoised = cv2.bilateralFilter(l_enhanced, 9, 75, 75)
#
#     merged = cv2.merge([denoised, a_channel, b_channel])
#     return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
#
# def process_faces(frame):
#     """Process faces with enhanced alignment, preprocessing, and scoring"""
#     face_results = yolo_model.predict(frame, conf=0.7)
#     best_label = "UNKNOWN"
#     best_bbox = []
#     best_score = 0.0
#     frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
#
#     for result in face_results:
#         for bbox in result.boxes.xyxy:
#             x1, y1, x2, y2 = map(int, bbox[:4])
#
#             # Skip small faces (minimum 100x100 pixels)
#             if (x2 - x1) < 100 or (y2 - y1) < 100:
#                 continue
#
#             try:
#                 cropped_face = frame[y1:y2, x1:x2]
#
#                 # Enhanced preprocessing
#                 aligned_face = align_face(cropped_face)
#                 enhanced_face = enhance_face_quality(aligned_face)
#
#                 # Get recognition score with debug
#                 raw_score, name = recognize_face(enhanced_face)
#                 print(f"Raw recognition score: {raw_score:.4f}")  # Debug output
#
#                 # Calculate position score
#                 face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
#                 distance = np.sqrt((face_center[0] - frame_center[0]) ** 2 +
#                                    (face_center[1] - frame_center[1]) ** 2)
#                 position_score = 1 / (1 + distance / 100)
#
#                 # Combined scoring
#                 composite_score = raw_score * 0.8 + position_score * 0.2
#
#                 # Adaptive threshold (lower for centered faces)
#                 dynamic_threshold = 0.52 if distance < 50 else 0.58
#
#                 if composite_score > best_score and composite_score >= dynamic_threshold:
#                     best_label = name
#                     best_bbox = [x1, y1, x2, y2]
#                     best_score = composite_score
#
#             except Exception as e:
#                 print(f"Face processing error: {e}")
#                 continue
#
#     return best_label, [best_bbox] if best_label != "UNKNOWN" else []