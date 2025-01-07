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
from Face_recognition.face_rec_retina_test import recognize_faces_in_persons
from ID_detection.yolov11.ID_Detection_test import detect_id_card
from Detection.Detection.settings import STATIC_ROOT
from Person_detection.Person_detection import detect_person

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

            if frame_count % 1 == 0:
                # Process the frame for person detection
                modified_frame, person_boxes = detect_person(frame)
                # Process faces and ID cards separately for each person
                people_data = []
                for person_box in person_boxes:
                    # Crop each person's image from the frame
                    x1, y1, x2, y2 = [int(coord) for coord in person_box]
                    person_image = frame[y1:y2, x1:x2]

                    # Detect face and ID card
                    person_flags, associations = recognize_faces_in_persons(person_image)
                    person_id_card_status = detect_id_card(person_image)

                    # Construct a person object (dictionary)
                    person = {
                        'bbox': person_box,
                        'face_flag': person_flags,
                        'id_card_status': person_id_card_status
                    }
                    people_data.append(person)

                # Save the detected information into MongoDB
                save_detections(people_data, camera_data[camera_index]['camera_location'])

            # Encode the frame to JPEG and then base64
            _, jpeg = cv2.imencode('.jpg', frame)
            current_frames[camera_index] = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        else:
            print(f"Error capturing frame from camera {camera_index}")


# Save the detection results to MongoDB
def save_detections(people_data, camera_location):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    for person in people_data:
        image_name = f"person_{camera_location}_{current_time}.jpg"
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)

        # Save person image to file
        os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)
        cv2.imwrite(image_path, person['bbox'])

        # Check if a document with the same person_id exists in the database
        person_id = tuple(person['bbox'])  # For simplicity, using bounding box as ID
        existing_document = collection.find_one({"person_id": person_id})

        if existing_document:
            # If the document exists, delete the old entry
            collection.delete_one({"_id": existing_document["_id"]})
            print(f"Deleted old entry with _id: {existing_document['_id']}")

        # Insert the new document
        document = {
            "_id": ObjectId(),
            "location": camera_location,
            "time": datetime.now().strftime("%D %I:%M %p"),
            "Role": "Unidentified" if person['face_flag'] == "UNKNOWN" else "SIETIAN",
            "Wearing_id_card": person['id_card_status'],
            "image": "/images/" + image_name,
            "person_id": person_id
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
