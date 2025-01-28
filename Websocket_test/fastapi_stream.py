import base64
import cv2
import json
import os
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from pathlib import Path
from Detection.Detection.settings import STATIC_ROOT
from Person_detection.Person_detection import track_persons
from Face_recognition.face_recognize_yolo import process_faces
from ID_detection.yolov11.ID_Detection_test import detect_id_card


IMAGE_FOLDER_PATH = os.path.join(STATIC_ROOT, 'images')

app = FastAPI()

# MongoDB connection setup
try:
    client = MongoClient(
        os.getenv("MONGO_URI", 'mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/'))
    db = client['ML_project']
    collection = db['DatabaseDB']
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit(1)

# Load camera data from data.json
DATA_FILE_PATH = Path('./Detection/data.json')
cached_data = None


def load_data():
    global cached_data
    if cached_data is None:
        if os.path.exists(DATA_FILE_PATH):
            try:
                with open(DATA_FILE_PATH, 'r') as file:
                    cached_data = json.load(file)
            except json.JSONDecodeError:
                print("Error: Failed to decode JSON from the file.")
                cached_data = []
        else:
            print("Error: File does not exist.")
            cached_data = []
    return cached_data


# Load camera data
camera_data = load_data()
print(camera_data)

# A dictionary to store frames for each camera
current_frames = {}

# ThreadPoolExecutor to handle concurrent frame processing
executor = ThreadPoolExecutor(max_workers=4)
def draw_annotations(frame, person_data):
    """Draw bounding boxes and annotations on the frame."""
    for person in person_data:
        x1, y1, x2, y2 = person['bbox']
        track_id = person['track_id']
        face_flag = person['face_flag']
        id_card = person['id_card']

        # Draw the person's bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare the text message
        text = f"ID: {track_id} | Face: {face_flag} | IDCard: {id_card}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Add a white background for better text visibility
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (255, 255, 255), -1)

        # Write the text above the bounding box
        cv2.putText(
            frame,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )


# Process a single frame (for each camera stream)
def process_frame(camera_index, camera_ip):
    cap = cv2.VideoCapture(camera_ip)  # RTSP stream URL

    if not cap.isOpened():
        print(f"Failed to open camera {camera_index} at {camera_ip}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            if frame_count % 3 == 0:
                person_results = track_persons(frame)

                # Check if person_results contains valid data
                if not person_results or "modified_frame" not in person_results or "person_boxes" not in person_results:
                    print("Invalid person results")
                    continue

                # frame = person_results["modified_frame"]
                person_boxes = person_results["person_boxes"]
                track_ids = person_results["track_ids"]
                people_data = []
                for person_box, track_id in zip(np.array(person_boxes).tolist(), track_ids):
                    x1, y1, x2, y2 = [int(coord) for coord in person_box]

                    # Validate and clip bounding boxes
                    frame_height, frame_width, _ = frame.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

                    # Crop the person image
                    person_image = frame[y1:y2, x1:x2]
                    if frame is None or person_image is None:
                        print(f"Empty or invalid image for track_id: {track_id}")
                        continue

                    # Initialize person data
                    person = {
                        'bbox': [x1, y1, x2, y2],
                        'track_id': track_id,
                        'face_flag': "UNKNOWN",
                        'face_box': [0, 0, 0, 0],
                        'id_flag': False,
                        'id_card': 'none',
                        'id_box': [0, 0, 0, 0]
                    }

                    # Face recognition
                    try:
                        person_flag, face_box = process_faces(frame)
                        person['face_flag'] = person_flag
                        person['face_box'] = face_box
                        print(f"Face recognition successful for track_id {track_id}")
                    except Exception as e:
                        print(f"Error during face recognition for track_id {track_id}: {e}")

                    # ID card detection (uncomment if you want to enable this)
                    try:
                        id_flag, id_box, id_card = detect_id_card(person_image)
                        person['id_flag'] = id_flag
                        person['id_box'] = id_box
                        person['id_card'] = id_card
                        print(f"ID card detection successful for track_id {track_id}")
                    except Exception as e:
                        print(f"Error during ID card detection for track_id {track_id}: {e}")

                    # Append person data
                    people_data.append(person)

                # Draw bounding boxes and annotations
                draw_annotations(frame, people_data)

                save_detections(people_data, camera_data[camera_index]['camera_location'])
                print("frame saved")
            # Encode the frame to JPEG and then base64
            _, jpeg = cv2.imencode('.jpg', frame)
            current_frames[camera_index] = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        else:
            print(f"Error capturing frame from camera {camera_index}")


def save_detections(people_data, camera_location):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    for person in people_data:
        image_name = f"person_{camera_location}_{current_time}.jpg"
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)

        os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)
        cv2.imwrite(image_path, person['bbox'])

        existing_document = collection.find_one({"track_id": person['track_id']})

        if existing_document:
            # If the document exists, delete the old entry
            collection.delete_one({"_id": existing_document["_id"]})
            print(f"Deleted old entry with _id: {existing_document['_id']}")

        document = {
            "_id": ObjectId(),
            "location": camera_location,
            "time": datetime.now().strftime("%D %I:%M %p"),
            "Role": "Unidentified" if person['face_flag'] == "UNKNOWN" else "SIETIAN",
            "Wearing_id_card": person['id_card_status'],
            "image": "/images/" + image_name,
            "track_id": person['track_id']
        }
        result = collection.insert_one(document)
        print(f"Document inserted with _id: {result.inserted_id}")


# Start separate threads for each camera
for index, camera in enumerate(camera_data):
    executor.submit(process_frame, index, camera["camera_ip"])


# WebSocket endpoint for video feed
@app.websocket("/ws/video/{camera_id}/")
async def video_feed(websocket: WebSocket, camera_id: int):
    await websocket.accept()

    while True:
        if camera_id in current_frames:
            # Decode the current frame from base64 to an image
            frame_data = base64.b64decode(current_frames[camera_id])
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            # Send the modified frame to the client as base64
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            await websocket.send_text(f'{{"frame": "{frame_b64}"}}')

            await asyncio.sleep(0.1)  # Sleep to prevent excessive CPU usage


if __name__ == "__main__":
    # Run the FastAPI server
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7000)
