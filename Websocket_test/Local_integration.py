from Face_recognition.face_recognize_yolo  import recognize_faces_in_persons
from ID_detection.yolov11.ID_Detection import detect_id_card
import os
import cv2
import time
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
import sys
import numpy as np

# Sensitive information as environment variables (better security practice)
username = os.getenv("CAMERA_USERNAME", "")
password = os.getenv("CAMERA_PASSWORD", "")
camera_ip = os.getenv("CAMERA_IP", "")
port = "554"  # Default RTSP port for Hikvision cameras

# Construct the RTSP URL
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:{port}/Streaming/Channels/101"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Face_rec-ID_detection')))

# MongoDB connection
try:
    client = MongoClient(os.getenv("MONGO_URI", 'mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/'))
    db = client['ML_project']  # Replace with your database name
    collection = db['DatabaseDB']  # Replace with your collection name
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    sys.exit(1)  # Exit if database connection fails


def process_and_save_detections(frame, person_bboxes, flags, associations, camera_id):
    """
    Process detections and save to MongoDB if abnormalities are detected.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, (person_box, flag) in enumerate(zip(person_bboxes, flags)):
        if flag == "UNDETERMINED":  # Skip if no face detected
            continue

        # Extract person sub-image from frame
        x1, y1, x2, y2 = [int(coord) for coord in person_box]
        person_image = frame[y1:y2, x1:x2]

        # Determine ID card association
        id_card_type = associations[idx] if idx < len(associations) else None
        wearing_id_card = bool(id_card_type)

        # Save data for specific conditions
        if flag in ["UNKNOWN", "SIETIAN"] and not wearing_id_card:
            image_name = f"person_{camera_id}_{current_time}_{idx}.jpg"
            image_path = os.path.join("images", image_name)

            try:
                os.makedirs("images", exist_ok=True)  # Ensure directory exists
                cv2.imwrite(image_path, person_image)

                document = {
                    "_id": ObjectId(),
                    "Reg_no": idx,
                    "location": camera_id,
                    "time": datetime.now(),
                    "Role": "Unidentified" if flag == "UNKNOWN" else "Student",
                    "Wearing_id_card": wearing_id_card,
                    "image": image_path,
                    "recognition_status": "Unknown" if flag == "UNKNOWN" else "Recognized",
                }

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
    save_interval = 2.0  # Save interval in seconds
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to capture image.")
                break

            frame_count += 1

            # Process every 5th frame
            if frame_count % 3 == 0:
                modified_frame, person_boxes, associations = detect_id_card(frame)
                modified_frame, flags = recognize_faces_in_persons(modified_frame, person_boxes)

                # Save detections periodically
                if time.time() - last_save_time >= save_interval:
                    process_and_save_detections(
                        frame=frame,
                        person_bboxes=person_boxes,
                        flags=flags,
                        associations=associations,
                        camera_id=camera_id
                    )
                    last_save_time = time.time()

                # Display the processed frame
                cv2.imshow('Video Feed', modified_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nExiting video feed...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def frame_test(frame):
    """
    Test frame processing for debugging.
    """
    modified_frame, person_boxes, associations = detect_id_card(frame)
    modified_frame, flags = recognize_faces_in_persons(modified_frame, person_boxes)
    cv2.imwrite('output_frame.jpg', modified_frame)


if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)  # Create directory for images if it doesn't exist
    video_feed()
    # frame_test("/run/media/drackko/022df0a1-27b0-4c14-ad57-636776986ded/drackko/PycharmProjects/Face_rec-ID_detection/WebApp/Detection/Face_recognition/test.jpg")