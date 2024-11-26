import cv2
import time
from ultralytics import YOLO

import os

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the YOLO model with the trained weights (relative path)
model_path = os.path.join(BASE_DIR,"models", "person+id", "best.pt")

model = YOLO(model_path)
# Define colors for different classes
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (120, 120, 0)]

# Function to check if two boxes overlap
def boxes_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

# Tracking data structure
tracked_persons = {}  # Key: Person ID, Value: {bounding_box, start_time, id_card_class}

def detect_id_card(frame):
    global tracked_persons

    results = model(frame)  # Perform inference with YOLO model

    current_time = time.time()
    persons = []
    id_cards = []

    # Loop over each detected object in the results
    for result in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        class_id = int(result.cls)
        class_name = results[0].names[class_id]
        confidence = result.conf.item()

        # Store detected persons and ID cards
        if class_name == 'Person':
            persons.append((x1, y1, x2, y2))
        elif class_name in ['II-id', 'III-id']:  # Assuming these are your ID card classes
            id_cards.append((x1, y1, x2, y2, class_name))

        # Draw bounding boxes for all detections
        label = f"{class_name} ({confidence:.2f})"
        color = colors[class_id % len(colors)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)

    # Associate ID cards with persons
    updated_tracked_persons = {}

    for person in persons:
        person_x1, person_y1, person_x2, person_y2 = person
        associated_id_card = None

        for id_card in id_cards:
            id_x1, id_y1, id_x2, id_y2, id_class_name = id_card

            if boxes_overlap((id_x1, id_y1, id_x2, id_y2), (person_x1, person_y1, person_x2, person_y2)):
                associated_id_card = id_class_name
                break

        # Check if this person was already being tracked
        for person_id, data in tracked_persons.items():
            prev_box, start_time, prev_id_card = data
            if boxes_overlap(prev_box, person):
                # Update tracking info
                updated_tracked_persons[person_id] = (
                    person,
                    start_time,
                    associated_id_card if associated_id_card else prev_id_card,
                )
                break
        else:
            # Start tracking new person
            updated_tracked_persons[len(updated_tracked_persons)] = (person, current_time, associated_id_card)

    # Determine final status for each tracked person
    association_status = []

    for person_id, data in updated_tracked_persons.items():
        box, start_time, id_card = data
        if current_time - start_time >= 3:  # Check if tracked for at least 3 seconds
            if id_card:
                association_status.append(f"{id_card}")
            else:
                association_status.append("no-id")
        else:
            association_status.append(f"Waiting for confirmation...")

        # Draw updated bounding boxes with status
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 255) if id_card else (0, 0, 255)
        cv2.putText(frame, association_status[-1], (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update tracked persons for the next frame
    tracked_persons = updated_tracked_persons

    return frame, persons, association_status
#
# if __name__ == "__main__":
#     # cap = cv2.VideoCapture()  # Replace with video file if needed
#     # while cap.isOpened():
#     #     ret, frame = cap.read()
#     #     if not ret:
#     #         break
#     #
#     #     annotated_frame = process_frame(frame)
#     #     cv2.imshow("Face Recognition", annotated_frame)
#     #
#     #     if cv2.waitKey(1) & 0xFF == ord("q"):
#     #         break
#     #
#     # cap.release()
#     img = cv2.imread("/run/media/drackko/022df0a1-27b0-4c14-ad57-636776986ded/drackko/PycharmProjects/Face_rec-ID_detection/WebApp/Detection/Face_recognition/test.jpg")
#     annotated_frame,_,_ = detect_id_card(img)
#     cv2.imwrite("op.jpg",annotated_frame)
#
