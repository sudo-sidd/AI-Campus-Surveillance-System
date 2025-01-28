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
from Face_recognition.face_recognize_yolo import recognize_faces_in_persons
from ID_detection.yolov11.ID_Detection_test import detect_id_card
from Detection.Detection.settings import STATIC_ROOT
from Person_detection.Person_detection import track_persons

IMAGE_FOLDER_PATH = os.path.join(STATIC_ROOT, 'images')

app = FastAPI()

try:
    client = MongoClient(
        os.getenv("MONGO_URI", 'mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/')
    )
    db = client['ML_project']
    collection = db['DatabaseDB']
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit(1)

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

camera_data = load_data()
print(camera_data)

current_frames = {}
executor = ThreadPoolExecutor(max_workers=4)

def process_frame(camera_index, camera_ip):
    cap = cv2.VideoCapture(camera_ip)

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
                frame = person_results["modified_frame"]
                person_boxes = person_results["person_boxes"]
                track_ids = person_results["track_ids"]

                people_data = []
                for person_box, track_id in zip(person_boxes, track_ids):

                    x1, y1, x2, y2 = [int(coord) for coord in person_box]
                    person_image = frame[y1:y2, x1:x2]

                    person_flags, associations = recognize_faces_in_persons(
                        person_image, person_box, track_id
                    )
                    person_id_card_status = detect_id_card(person_image)

                    person = {
                        'bbox': person_box,
                        'track_id': track_id,
                        'face_flag': person_flags,
                        'id_card_status': person_id_card_status
                    }
                    people_data.append(person)

                save_detections(people_data, camera_data[camera_index]['camera_location'])

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

        person_id = tuple(person['bbox'])
        existing_document = collection.find_one({"track_id": person['track_id']})

        if existing_document:
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

for index, camera in enumerate(camera_data):
    executor.submit(process_frame, index, camera["camera_ip"])

@app.websocket("/ws/video/{camera_id}/")
async def video_feed(websocket: WebSocket, camera_id: int):
    await websocket.accept()

    while True:
        if camera_id in current_frames:

            frame_data = base64.b64decode(current_frames[camera_id])
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            _, jpeg = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            await websocket.send_text(f'{{"frame": "{frame_b64}"}}')

            await asyncio.sleep(0.1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
