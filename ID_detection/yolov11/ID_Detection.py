import cv2
from ultralytics import YOLO

# Load the YOLO model with the trained weights
model = YOLO("ID_detection/yolov11/models/person+id/best.pt")  # Adjust this to the correct model path

# Define colors for different classes
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (120, 120, 0)]

# Function to check if two boxes overlap
def boxes_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

def detect_id_card(frame):
    results = model(frame)  # Perform inference with YOLO model

    persons = []
    id_cards = []
    association_status = []  # List to store association statuses

    # Loop over each detected object in the results
    for result in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        class_id = int(result.cls)
        class_name = results[0].names[class_id]
        confidence = result.conf.item()

        # Store detected persons and ID cards
        if class_name == 'person':
            persons.append((x1, y1, x2, y2))
            association_status.append("No ID card associated")  # Default status for persons
        elif class_name in ['II-id', 'III-id']:  # Assuming these are your ID card classes
            id_cards.append((x1, y1, x2, y2, class_name))

        # Draw bounding boxes for all detections (optional)
        label = f"{class_name} ({confidence:.2f})"
        color = colors[class_id % len(colors)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)

    # Associate ID cards with persons based on proximity
    for i, person in enumerate(persons):
        person_x1, person_y1, person_x2, person_y2 = person
        associated_id_card = False

        for id_card in id_cards:
            id_x1, id_y1, id_x2, id_y2, id_class_name = id_card

            if boxes_overlap((id_x1, id_y1, id_x2, id_y2), (person_x1, person_y1, person_x2, person_y2)):
                # If they overlap or are close enough
                association_status[i] = f"ID Card: {id_class_name} associated with Person"
                associated_id_card = True

        if not associated_id_card:
            association_status[i] = "No ID card associated with Person"

    return frame, persons, association_status