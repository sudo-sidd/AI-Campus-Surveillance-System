import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from ultralytics import YOLO
from .face_recognition.LightCNN.light_cnn import LightCNN_29Layers_v2
from PIL import Image
from collections import deque, Counter
from .face_alignment.alignment import get_landmarks, align_face

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize YOLO face detection model
yolo_model = YOLO(os.path.join(BASE_DIR, "face_detection", "yolo", "weights", "yolo11n-face.pt"))

# Initialize LightCNN model
model_weights_path = os.path.join(BASE_DIR, "face_recognition", "LightCNN", "model", "test.pth")
class_names = ["mithun","sai"]
recognizer = None

try:
    # Try loading with state_dict format first
    checkpoint = torch.load(model_weights_path, map_location=device)
    
    # Get class mapping from checkpoint
    if 'idx_to_class' in checkpoint:
        idx_to_class = checkpoint['idx_to_class']
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        print(f"Loaded {len(class_names)} classes from checkpoint: {class_names}")
    
    # Initialize model with correct number of classes
    num_classes = len(class_names)
    recognizer = LightCNN_29Layers_v2(num_classes=num_classes).to(device)
    
    # Load weights
    if 'state_dict' in checkpoint:
        recognizer.load_state_dict(checkpoint['state_dict'])
    else:
        # Fallback to direct loading
        recognizer.load_state_dict(checkpoint)
    
    recognizer.eval()
    print(f"Successfully loaded LightCNN model from {model_weights_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Recognition thresholds and parameters (MODIFIED VALUES)
CONFIDENCE_THRESHOLD = 0.45  # REDUCED: Allow lower confidence matches
RECOGNITION_THRESHOLD = 0.7   # Higher threshold for definitive match
HISTORY_SIZE = 15  # Number of frames to keep in history
CONSISTENCY_THRESHOLD = 0.5  # REDUCED: Less strict consistency requirement
UNKNOWN_THRESHOLD = 0.8  # INCREASED: More unknowns required to reset
MIN_FRAMES_FOR_DECISION = 5  # REDUCED: Faster decisions

# Recognition memory for temporal consistency
recognition_history = {}  # track_id -> recognition history

def preprocess_face(face_image):
    """Preprocess face image for LightCNN - improved grayscale handling"""
    try:
        # First, ensure we have a valid image
        if face_image is None or face_image.size == 0:
            print("Warning: Empty face image in preprocessing")
            # Return a placeholder tensor
            return torch.zeros((1, 1, 128, 128), device=device)
        
        # Apply CLAHE to enhance contrast before grayscale conversion
        lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Convert to grayscale with OpenCV
        gray_face = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        face_pil = Image.fromarray(gray_face)
        
        # Use transforms without redundant grayscale conversion
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        return transform(face_pil).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error in preprocess_face: {e}")
        return torch.zeros((1, 1, 128, 128), device=device)

@torch.no_grad()
def recognize_face(face_image):
    """Recognize a face using LightCNN"""
    if recognizer is None:
        return 0.0, "UNKNOWN", []
        
    try:
        # Preprocess the face image
        face_tensor = preprocess_face(face_image)
        
        # Get embedding and predictions from model
        output = recognizer(face_tensor)
        
        # Handle output format (features, logits)
        if isinstance(output, tuple):
            _, logits = output
        else:
            logits = output
            
        # Get top 3 predictions with confidences
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        top_values, top_indices = torch.topk(probs, min(3, len(probs)))
        
        # Get all predictions
        predictions = []
        for i in range(len(top_indices)):
            idx = top_indices[i].item()
            conf = top_values[i].item()
            if idx < len(class_names):
                predictions.append((class_names[idx], conf, idx))
                # Debug output for top prediction
                if i == 0:
                    print(f"Top prediction: {class_names[idx]} with confidence {conf:.4f}")
        
        # If no valid predictions, return unknown
        if not predictions:
            return 0.0, "UNKNOWN", []
            
        return predictions[0][1], predictions[0][0], predictions
    except Exception as e:
        print(f"Recognition error: {e}")
        return 0.0, "UNKNOWN", []

def update_recognition_memory(track_id, new_name, new_score):
    """Update recognition memory for temporal consistency"""
    if track_id not in recognition_history:
        recognition_history[track_id] = {
            "history": deque(maxlen=HISTORY_SIZE),
            "current_label": "UNKNOWN",
            "frames_since_known": 0,
            "frames_seen": 0
        }
    
    memory = recognition_history[track_id]
    memory["history"].append((new_name, new_score))
    memory["frames_seen"] += 1
    
    # Record all detections for debugging
    print(f"Track ID {track_id}: New detection [{new_name}] with score {new_score:.4f}")
    
    # If we have a high confidence match, immediately accept it
    if new_score >= RECOGNITION_THRESHOLD and new_name != "UNKNOWN":
        memory["current_label"] = new_name
        memory["frames_since_known"] = 0
        print(f"High confidence match! Setting track {track_id} to {new_name}")
        return memory["current_label"]
    
    # If we have enough history, make decisions based on it
    if len(memory["history"]) >= MIN_FRAMES_FOR_DECISION:
        # Get names with their counts
        names = [entry[0] for entry in memory["history"]]
        name_counts = Counter(names)
        
        # Find the most common name
        most_common_name, count = name_counts.most_common(1)[0]
        
        # Calculate the proportion of this name in the history
        proportion = count / len(memory["history"])
        
        # If UNKNOWN appears too frequently, maintain UNKNOWN status
        unknown_count = sum(1 for name, _ in memory["history"] if name == "UNKNOWN")
        unknown_proportion = unknown_count / len(memory["history"])
        
        print(f"Track {track_id} history stats: Most common={most_common_name} ({proportion:.2f}), Unknown={unknown_proportion:.2f}")
        
        if unknown_proportion >= UNKNOWN_THRESHOLD:
            memory["frames_since_known"] += 1
            
            # Only change to UNKNOWN after consecutive frames
            if memory["frames_since_known"] > MIN_FRAMES_FOR_DECISION:
                if memory["current_label"] != "UNKNOWN":
                    print(f"Resetting track {track_id} from {memory['current_label']} to UNKNOWN")
                memory["current_label"] = "UNKNOWN"
        
        # If we have a consistent non-UNKNOWN name, use it
        elif most_common_name != "UNKNOWN" and proportion >= CONSISTENCY_THRESHOLD:
            if memory["current_label"] != most_common_name:
                print(f"Updating track {track_id} from {memory['current_label']} to {most_common_name}")
            memory["current_label"] = most_common_name
            memory["frames_since_known"] = 0
    
    return memory["current_label"]

def process_faces(frame, track_id=None):
    """
    Enhanced face processing with improved multi-person handling.
    
    Args:
        frame: The image frame to process
        track_id: Optional track ID for temporal consistency
        
    Returns:
        tuple: (best_label, best_score, [best_bbox])
    """
    try:
        # Verify input frame
        if frame is None or frame.size == 0:
            print("Warning: Empty frame in process_faces")
            return ("UNKNOWN", 0.0, [])
            
        # Get original frame dimensions for debug output
        frame_height, frame_width = frame.shape[:2]
        print(f"Processing frame {frame_width}x{frame_height} for track {track_id}")
            
        face_results = yolo_model.predict(frame,imgsz=frame.shape[:2], conf=0.4)  # Lowered confidence threshold
        
        # Create a list of all detected faces
        detected_faces = []

        # First pass: Detect all faces and recognize them
        for result in face_results:
            for bbox in result.boxes.xyxy:
                try:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    
                    # Initial face detection data
                    face_data = {
                        'bbox': [x1, y1, x2, y2],
                        'area': (x2 - x1) * (y2 - y1),
                        'name': "UNKNOWN",
                        'confidence': 0.0,
                        'predictions': []
                    }
                    
                    print(f"Detected face at [{x1},{y1},{x2},{y2}], size: {x2-x1}x{y2-y1}")
                    
                    # Skip very small faces (reduced minimum size)
                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        print(f"Skipping small face: {x2-x1}x{y2-y1}")
                        detected_faces.append(face_data)
                        continue
                    
                    # Extract face with a small margin
                    margin_x = int((x2 - x1) * 0.1)
                    margin_y = int((y2 - y1) * 0.1)
                    y1_safe = max(0, y1 - margin_y)
                    y2_safe = min(frame.shape[0], y2 + margin_y)
                    x1_safe = max(0, x1 - margin_x)
                    x2_safe = min(frame.shape[1], x2 + margin_x)
                    
                    cropped_face = frame[y1_safe:y2_safe, x1_safe:x2_safe]
                    
                    if cropped_face.size == 0:
                        print("Empty face region after cropping")
                        detected_faces.append(face_data)
                        continue
                    
                    # Try both with and without alignment
                    try:
                        # First try with alignment
                        landmarks = get_landmarks(cropped_face)
                        aligned_face = align_face(cropped_face, landmarks, output_size=128)
                        confidence, name, predictions = recognize_face(aligned_face)
                    except Exception as align_err:
                        print(f"Face alignment failed: {align_err}, trying without alignment")
                        # If alignment fails, try without it
                        confidence, name, predictions = recognize_face(cropped_face)
                    
                    # Store recognition results
                    face_data['name'] = name
                    face_data['confidence'] = confidence
                    face_data['predictions'] = predictions
                    detected_faces.append(face_data)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    # Add face with default values
                    detected_faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'area': (x2 - x1) * (y2 - y1),
                        'name': "UNKNOWN",
                        'confidence': 0.0,
                        'predictions': []
                    })
        
        # If there are no detected faces, return default values
        if not detected_faces:
            print("No faces detected in frame")
            return ("UNKNOWN", 0.0, [])
        
        # Second pass: Resolve identity conflicts by assigning each class to exactly one face
        # For each class, only the face with highest confidence gets that label
        class_to_best_face = {}  # Maps class name to (face_index, confidence)
        
        for face_idx, face_data in enumerate(detected_faces):
            for prediction in face_data['predictions']:
                if len(prediction) < 3:
                    continue  # Skip invalid predictions
                    
                class_name, confidence, _ = prediction
                
                # Skip unknown class for assignment
                if class_name == "UNKNOWN":
                    continue
                    
                # Skip very low confidence predictions
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                    
                # Is this the best confidence we've seen for this class?
                if (class_name not in class_to_best_face or 
                    confidence > class_to_best_face[class_name][1]):
                    class_to_best_face[class_name] = (face_idx, confidence)
                    print(f"Best match for {class_name}: face {face_idx} with confidence {confidence:.4f}")
        
        # Third pass: Apply final classifications to faces
        for face_idx, face_data in enumerate(detected_faces):
            assigned_label = "UNKNOWN"
            assigned_confidence = 0.0
            
            # Check if this face is the best match for any class
            for class_name, (best_face_idx, confidence) in class_to_best_face.items():
                if face_idx == best_face_idx and class_name != "UNKNOWN":
                    assigned_label = class_name
                    assigned_confidence = confidence
                    break
            
            # Store the final assignment
            face_data['name'] = assigned_label
            face_data['confidence'] = assigned_confidence
        
        # If track_id is provided, we need to determine which face belongs to this track
        if track_id is not None:
            # Sort faces by area (largest first) as a heuristic
            detected_faces.sort(key=lambda x: x['area'], reverse=True)
            largest_face = detected_faces[0]  # Assume the largest face corresponds to this track_id
            raw_label = largest_face['name']
            raw_score = largest_face['confidence']
            best_bbox = largest_face['bbox']
            
            print(f"Track {track_id} - Raw recognition: {raw_label} ({raw_score:.4f})")
            
            # Apply temporal consistency
            best_label = update_recognition_memory(track_id, raw_label, raw_score)
            return (best_label, raw_score, [best_bbox])
        
        # If no track_id (processing faces in isolation):
        # Sort by confidence*area to prioritize confident and large faces
        detected_faces.sort(key=lambda x: x['confidence'] * x['area'], reverse=True)
        
        # Return the best face
        best_face = detected_faces[0]
        return (best_face['name'], best_face['confidence'], [best_face['bbox']])
        
    except Exception as e:
        print(f"Unexpected error in process_faces: {e}")
        import traceback
        traceback.print_exc()
        return ("UNKNOWN", 0.0, [])