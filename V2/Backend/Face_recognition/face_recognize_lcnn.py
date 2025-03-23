import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from ultralytics import YOLO
from .face_recognition.LightCNN.light_cnn import LightCNN_29Layers_v2
from PIL import Image
import time
from collections import deque, Counter

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

# Recognition history for temporal smoothing
recognition_history = {}  # track_id -> deque of recent recognitions
HISTORY_SIZE = 15  # Store more history frames
CONSISTENCY_THRESHOLD = 0.6  # Minimum consistency required
CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence to consider
UNKNOWN_THRESHOLD = 0.7  # Proportion of frames needed to declare UNKNOWN
MIN_FRAMES_FOR_DECISION = 5  # Minimum frames before making a decision

# Face recognition memory
face_recognition_memory = {}

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
    """Improved face enhancement with color preservation"""
    try:
        # Convert to LAB color space to separate luminance from color
        lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE only to the luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # Merge back the enhanced luminance with original color
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Additional noise reduction
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
        
        return enhanced
    except Exception as e:
        print(f"Enhancement error: {e}")
        return face_image  # Return original if enhancement fails


# Modify process_faces() to accept track_id parameter
def process_faces(frame, track_id=None):
    """Enhanced face processing with temporal consistency"""
    face_results = yolo_model.predict(frame, conf=0.5)
    
    best_label = "UNKNOWN"
    best_bbox = []
    best_score = 0.0
    
    all_faces = []  # Store all recognized faces for debugging
    
    # Process detected faces
    for result in face_results:
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # More permissive size filter (60x60 instead of 100x100)
            if (x2 - x1) < 60 or (y2 - y1) < 60:
                continue
                
            try:
                # Extract face with a small margin
                margin_x = int((x2 - x1) * 0.1)
                margin_y = int((y2 - y1) * 0.1)
                y1_safe = max(0, y1 - margin_y)
                y2_safe = min(frame.shape[0], y2 + margin_y)
                x1_safe = max(0, x1 - margin_x)
                x2_safe = min(frame.shape[1], x2 + margin_x)
                
                cropped_face = frame[y1_safe:y2_safe, x1_safe:x2_safe]
                
                if cropped_face.size == 0:
                    continue
                
                # Enhanced preprocessing pipeline
                aligned_face = align_face(cropped_face)
                enhanced_face = enhance_face_quality(aligned_face)
                
                # Get recognition confidence and label
                confidence, name = recognize_face(enhanced_face)
                
                # Use a flat confidence threshold rather than position-based
                if confidence >= 0.45:  # Lower base threshold
                    all_faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'name': name,
                        'confidence': confidence,
                        'area': (x2 - x1) * (y2 - y1)  # Use face area for ranking
                    })
            
            except Exception as e:
                print(f"Face processing error: {e}")
                continue
    
    # Sort faces by area (largest first) and then by confidence
    all_faces.sort(key=lambda x: (x['area'], x['confidence']), reverse=True)
    
    # If we have any recognized faces, take the most confident one
    if all_faces:
        face = all_faces[0]
        raw_label = face['name']
        raw_score = face['confidence']
        best_bbox = face['bbox']
        
        # Apply temporal consistency if track_id is provided
        if track_id is not None:
            # Initialize history for this track_id if it doesn't exist
            if track_id not in recognition_history:
                recognition_history[track_id] = deque(maxlen=HISTORY_SIZE)
                
            # Add current recognition to history
            recognition_history[track_id].append((raw_label, raw_score))
            
            # Count occurrences of each label in history
            label_counts = {}
            score_sums = {}
            
            for hist_label, hist_score in recognition_history[track_id]:
                if hist_label not in label_counts:
                    label_counts[hist_label] = 0
                    score_sums[hist_label] = 0
                label_counts[hist_label] += 1
                score_sums[hist_label] += hist_score
            
            # Find the most consistent label
            max_count = 0
            consistent_label = "UNKNOWN"
            consistent_score = 0
            
            for label, count in label_counts.items():
                consistency_ratio = count / len(recognition_history[track_id])
                avg_score = score_sums[label] / count
                
                if consistency_ratio >= CONSISTENCY_THRESHOLD and count > max_count:
                    max_count = count
                    consistent_label = label
                    consistent_score = avg_score
            
            # Use the consistent label if found, otherwise use the raw label
            best_label = consistent_label if consistent_label != "UNKNOWN" else raw_label
            best_score = consistent_score if consistent_label != "UNKNOWN" else raw_score
        else:
            # No track_id provided, use raw results
            best_label = raw_label
            best_score = raw_score
    
    # For debugging, print all detected faces and their scores
    if len(all_faces) > 0:
        debug_msg = "Detected faces: " + " | ".join(
            [f"{f['name']}:{f['confidence']:.2f}" for f in all_faces])
        print(debug_msg)
    
    return (best_label, best_score, [best_bbox])


def update_recognition_memory(track_id, new_name, new_score):
    """
    Maintain recognition memory for each tracked person with majority voting system.
    Returns the most consistent label based on recent history.
    """
    # Initialize if this is a new track_id
    if track_id not in face_recognition_memory:
        face_recognition_memory[track_id] = {
            "history": deque(maxlen=HISTORY_SIZE),
            "current_label": "UNKNOWN",
            "frames_since_known": 0
        }
    
    memory = face_recognition_memory[track_id]
    
    # Only add to history if confidence exceeds minimum threshold
    if new_score >= CONFIDENCE_THRESHOLD or new_name != "UNKNOWN":
        memory["history"].append((new_name, new_score))
    
    # If we have enough frames, make a decision based on majority voting
    if len(memory["history"]) >= MIN_FRAMES_FOR_DECISION:
        # Count occurrences of each name (excluding low confidence)
        valid_entries = [(name, score) for name, score in memory["history"] 
                         if score >= CONFIDENCE_THRESHOLD or name != "UNKNOWN"]
        
        if valid_entries:
            # Get names with their counts
            names = [entry[0] for entry in valid_entries]
            name_counts = Counter(names)
            
            # Find the most common name
            most_common_name, count = name_counts.most_common(1)[0]
            
            # Calculate the proportion of this name in the history
            proportion = count / len(memory["history"])
            
            # If UNKNOWN appears too frequently, maintain UNKNOWN status
            unknown_count = sum(1 for name, _ in memory["history"] if name == "UNKNOWN")
            unknown_proportion = unknown_count / len(memory["history"])
            
            if unknown_proportion >= UNKNOWN_THRESHOLD:
                memory["frames_since_known"] += 1
                
                # Only change to UNKNOWN after consecutive frames
                if memory["frames_since_known"] > MIN_FRAMES_FOR_DECISION:
                    memory["current_label"] = "UNKNOWN"
            
            # If we have a consistent non-UNKNOWN name, use it
            elif most_common_name != "UNKNOWN" and proportion >= 0.5:
                memory["current_label"] = most_common_name
                memory["frames_since_known"] = 0
        
    return memory["current_label"]