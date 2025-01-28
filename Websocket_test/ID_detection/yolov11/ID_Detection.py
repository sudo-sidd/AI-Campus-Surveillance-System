import cv2
import time
from ultralytics import YOLO
import os
import numpy as np

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the YOLO model with the trained weights (relative path)
model_path = os.path.join(BASE_DIR, "models", "id-model.pt")
model = YOLO(model_path)

# Define colors for bounding boxes
color = (0, 255, 0)

# Tracking data structure
tracked_items = {}  # Key: ID, Value: {bounding_box, start_time}

# Function to check if two boxes overlap
def boxes_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

# Function to apply Non-Maximum Suppression (NMS)
def apply_nms(boxes, confidences, threshold=0.5):
    # Convert boxes to np.array and perform NMS
    boxes = np.array(boxes)
    confidences = np.array(confidences)

    # Apply NMS (you may need to adjust the IoU threshold to suit your needs)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), score_threshold=0.4, nms_threshold=threshold)

    # Filter boxes based on NMS results
    filtered_boxes = []
    filtered_confidences = []
    if len(indices) > 0:
        for i in indices.flatten():
            filtered_boxes.append(boxes[i])
            filtered_confidences.append(confidences[i])

    return filtered_boxes, filtered_confidences

# Function to detect ID card
def detect_id_card(frame):
    global tracked_items

    results = model(frame)  # Perform inference with YOLO model
    current_time = time.time()
    detected_items = []
    boxes = []
    confidences = []

    # Loop over each detected object in the results
    for result in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        confidence = result.conf.item()

        # Only process if confidence is 75% or above
        if confidence >= 0.75:
            boxes.append([x1, y1, x2, y2])
            confidences.append(confidence)

    # Apply Non-Maximum Suppression to reduce overlapping
    filtered_boxes, filtered_confidences = apply_nms(boxes, confidences, threshold=0.5)

    # Draw filtered bounding boxes for detections
    for box, confidence in zip(filtered_boxes, filtered_confidences):
        x1, y1, x2, y2 = box
        label = f"ID Card ({confidence:.2f})"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)

    # Update tracked items
    updated_tracked_items = {}

    for item in filtered_boxes:
        item_x1, item_y1, item_x2, item_y2 = item

        # Check if this item was already being tracked
        for item_id, data in tracked_items.items():
            prev_box, start_time = data
            if boxes_overlap(prev_box, item):
                # Update tracking info
                updated_tracked_items[item_id] = (item, start_time)
                break
        else:
            # Start tracking new item
            updated_tracked_items[len(updated_tracked_items)] = (item, current_time)

    # Determine final status for each tracked item
    association_status = []

    for item_id, data in updated_tracked_items.items():
        box, start_time = data
        if current_time - start_time >= 2:  # Check if tracked for at least 2 seconds
            association_status.append("ID Card Detected")
        else:
            association_status.append("Waiting for confirmation...")

        # Draw updated bounding boxes with status
        x1, y1, x2, y2 = map(int, box)
        status_color = (0, 255, 255) if current_time - start_time >= 2 else (0, 0, 255)
        cv2.putText(frame, association_status[-1], (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # Update tracked items for the next frame
    tracked_items = updated_tracked_items

    return frame, association_status
#
# # Open webcam (camera ID 0)
# cap = cv2.VideoCapture(2)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame. Exiting...")
#         break
#
#     # Run detection on the frame
#     frame, association_status = detect_id_card(frame)
#
#     # Show the frame with bounding boxes and status
#     cv2.imshow("ID Card Detection", frame)
#
#     # Exit the loop when the user presses 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
