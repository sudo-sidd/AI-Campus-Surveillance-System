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
# Import the new face alignment module
from .face_alignment.alignment import align_face, enhance_image, get_landmarks

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize YOLO face detection model
yolo_model = YOLO(os.path.join(BASE_DIR, "face_detection", "yolo", "weights", "yolo11n-face.pt")).to(device)
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
MIN_FRAMES_FOR_DECISION = 8 # Minimum frames before making a decision

# Face recognition memory
face_recognition_memory = {}

def preprocess_face(face_image):
    """Preprocess face image for LightCNN - improved grayscale handling"""
    # Convert to grayscale with OpenCV (single conversion)
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_pil = Image.fromarray(gray_face)
    
    # Use transforms without redundant grayscale conversion
    transform = transforms.Compose([
        # No Grayscale here since already converted
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    return transform(face_pil).unsqueeze(0).to(device)

@torch.no_grad()
def recognize_face(face_image):
    """Recognize a face using LightCNN"""
    try:
        # Preprocess the face image
        face_tensor = preprocess_face(face_image)
        
        # Get embedding and predictions from model
        _, output = recognizer(face_tensor)
        
        # Get confidence scores
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, index = torch.max(probs, 0)
        confidence = confidence.item()
        
        # Get the corresponding name
        name = class_names[index] if confidence >= CONFIDENCE_THRESHOLD else "UNKNOWN"
        
        return confidence, name
    except Exception as e:
        print(f"Recognition error: {e}")
        return 0.0, "UNKNOWN"

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

# Modify process_faces() to accept track_id parameter
def process_faces(frame, track_id=None):
    """Enhanced face processing with improved alignment and temporal consistency"""
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
                
                # Enhanced preprocessing pipeline - using new alignment module
                # 1. Get landmarks if possible
                landmarks = get_landmarks(cropped_face)
                
                # 2. Align face using landmarks or use center crop as fallback
                aligned_face = align_face(cropped_face, landmarks, output_size=128)
                
                # 3. Enhance image quality
                # enhanced_face = enhance_image(aligned_face)
                
                # 4. Get recognition confidence and label
                confidence, name = recognize_face(aligned_face)
                
                # Use a flat confidence threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    all_faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'name': name,
                        'confidence': confidence,
                        'area': (x2 - x1) * (y2 - y1)
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
            best_label = update_recognition_memory(track_id, raw_label, raw_score)
            best_score = raw_score  # Use raw score since memory already includes confidence weighting
        else:
            best_label = raw_label
            best_score = raw_score
    
    return (best_label, best_score, [best_bbox])