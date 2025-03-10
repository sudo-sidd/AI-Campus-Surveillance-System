import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from ultralytics import YOLO
from .face_recognition.LightCNN.light_cnn import LightCNN_29Layers_v2
from PIL import Image

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize YOLO face detection model
yolo_model = YOLO(os.path.join(BASE_DIR, "face_detection", "yolo", "weights", "yolo11n-face.pt"))

# Initialize LightCNN model
class_names = ["sidd", "sai"]  # Load from file/database in production
model_weights_path = os.path.join(BASE_DIR, "face_recognition", "LightCNN", "model", "sidd_sai_fisrt.pth")
recognizer = LightCNN_29Layers_v2(num_classes=len(class_names)).to(device)
recognizer.load_state_dict(torch.load(model_weights_path, map_location=device))
recognizer.eval()

# Image preprocessing transforms
face_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def preprocess_face(face_image):
    """Preprocess face image for LightCNN"""
    face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY))
    return face_transform(face_pil).unsqueeze(0).to(device)


@torch.no_grad()
def recognize_face(face_image):
    """Recognize a face image using LightCNN"""
    try:
        face_tensor = preprocess_face(face_image)
        _, logits = recognizer(face_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)

        if 0 <= pred_idx < len(class_names):
            return confidence.item(), class_names[pred_idx]
        return 0.0, "UNKNOWN"
    except Exception as e:
        print(f"Recognition error: {e}")
        return 0.0, "UNKNOWN"


def align_face(face_image):
    """Center crop and resize face for better alignment"""
    h, w = face_image.shape[:2]
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    aligned = face_image[y_start:y_start + size, x_start:x_start + size]
    return cv2.resize(aligned, (128, 128))


def enhance_face_quality(face_image):
    """Improve face image quality using CLAHE (grayscale version)"""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert back to BGR for consistency
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def process_faces(frame):
    """Enhanced face processing with LightCNN integration"""
    face_results = yolo_model.predict(frame, conf=0.8)
    best_label = "UNKNOWN"
    best_bbox = []
    best_score = 0.0
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

    for result in face_results:
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox[:4])

            # Skip small detections (min 100x100 pixels)
            if (x2 - x1) < 100 or (y2 - y1) < 100:
                continue

            try:
                # Extract and preprocess face
                cropped_face = frame[y1:y2, x1:x2]
                aligned_face = align_face(cropped_face)
                enhanced_face = enhance_face_quality(aligned_face)

                # Get recognition score
                raw_score, name = recognize_face(enhanced_face)

                # Calculate position-based score
                face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                distance = np.sqrt((face_center[0] - frame_center[0]) ** 2 +
                                   (face_center[1] - frame_center[1]) ** 2)
                position_score = 1 / (1 + distance / 100)

                # Combine scores with weighted average
                composite_score = raw_score * 0.7 + position_score * 0.3

                # Adaptive threshold based on face position
                dynamic_threshold = 0.50 if distance < 150 else 0.65

                if composite_score > best_score and composite_score >= dynamic_threshold:
                    best_label = name
                    best_bbox = [x1, y1, x2, y2]
                    best_score = composite_score

            except Exception as e:
                print(f"Face processing error: {e}")
                continue

    return (best_label, best_score, [best_bbox])