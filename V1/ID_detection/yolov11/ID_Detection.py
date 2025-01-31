from ultralytics import YOLO
import os

# Set up paths and load YOLO model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "person+id", "best.pt")
model = YOLO(model_path)

# Define colors for bounding boxes
color = (0, 255, 0)

# Class names for the IDs
class_names = {0: "III-id", 1: "II-id", 2: "Person"}  # Adjust according to your model's class names

# Function to detect ID cards
def detect_id_card(frame):
    results = model(frame)
    id_card_detected = False  # Boolean to indicate if an ID card is detected
    bbox = []  # Bounding box of the detected ID card
    id_card_type = ""  # Type of the detected ID card
    confidence = 0.0  # Confidence score of the detected ID card

    for result in results[0].boxes:
        # Get confidence and class ID
        confidence = result.conf.item()
        class_id = int(result.cls.item())
        if confidence >= 0.55 and class_id in [0, 1]:  # III-id: 0, II-id: 1
            # Get the bounding box coordinates
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            id_card_type = class_names[class_id]
            id_card_detected = True
            break  # Exit the loop as we only need one ID card

    return id_card_detected, bbox, id_card_type