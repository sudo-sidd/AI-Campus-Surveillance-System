from Face_recognition.face_recognize import recognize_face
from ID_detection.yolov11.ID_Detection import detect_id_card
import os
import cv2
import time
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId

# Set up MongoDB client
client = MongoClient('mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/')
db = client['ML_project']  # Replace with your database name
collection = db['DatabaseDB']  # Replace with your collection name

def process_and_save_detections(frame, person_boxes, flags, associations, camera_id):
    """
    Process detections and save to MongoDB.

    Args:
        frame: The video frame.
        person_boxes: List of person bounding boxes.
        flags: List of recognition flags ("SIETIAN", "UNKNOWN", or "_").
        associations: ID card associations from detect_id_card.
        camera_id: Identifier for the camera.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, (person_box, flag) in enumerate(zip(person_boxes, flags)):
        if flag == "_":  # Skip if no face detected
            continue

        # Extract person sub-image from frame
        x1, y1, x2, y2 = [int(coord) for coord in person_box]
        person_image = frame[y1:y2, x1:x2]

        # Get ID card type from associations
        if idx < len(associations):  # Check if index is valid
            id_card_type = associations[idx]  # Use indexing instead of .get()
            wearing_id_card = bool(id_card_type)
        else:
            id_card_type = None  # Default value if out of range
            wearing_id_card = False  # Default value

        # Generate unique image name and save locally
        image_name = f"person_{camera_id}_{current_time}_{idx}.jpg"
        image_path = os.path.join("images", image_name)  # Ensure this directory exists

        try:
            cv2.imwrite(image_path, person_image)  # Save image locally

            # Prepare data for MongoDB based on recognition flags
            if flag.startswith("SIETIAN"):  # Known face recognized as SIETIAN
                person_name = flag.split(" ")[0]  # Extract name if needed (e.g., "SIETIAN (John Doe)")
                role = "Student"  # Or get from your recognition system
                recognition_status = "Recognized"
            else:  # UNKNOWN or other statuses
                person_name = "Unknown Person"
                role = "Unidentified"
                recognition_status = "Unknown"

            # Create document to insert into MongoDB
            document = {
                "_id": ObjectId(),  # Generate a random ObjectId
                "Reg_no": idx,  # You can modify this based on your requirements
                "location": camera_id,
                "time": datetime.now(),
                "Role": role,
                "Wearing_id_card": wearing_id_card,
                "image": image_path,
                "recognition_status": recognition_status,
            }

            # Insert document into MongoDB
            result = collection.insert_one(document)
            print(f"Document inserted with _id: {result.inserted_id}")

        except Exception as e:
            print(f"Error saving detection data to database: {e}")

def video_feed(camera_id=0):
    """
    Main video processing loop.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    last_save_time = time.time()
    save_interval = 2.0  # Save every 2 seconds

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image.")
            break

        # Get person boxes and ID card detections.
        modified_frame, person_boxes, associations = detect_id_card(frame)

        # Get face recognition results.
        modified_frame, flags = recognize_face(modified_frame, person_boxes)

        # Save detections periodically.
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            process_and_save_detections(
                frame=frame,
                person_boxes=person_boxes,
                flags=flags,
                associations=associations,
                camera_id=camera_id
            )
            last_save_time = current_time

        # Display the processed frame.
        cv2.imshow('Video Feed', modified_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)  # Create directory for images if it doesn't exist
    video_feed()
