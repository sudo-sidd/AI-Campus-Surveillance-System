import base64
import cv2
import json
import os
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket
import threading
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from pathlib import Path
from Face_recognition.face_recognize_yolo_updated import recognize_faces_in_persons
from Face_recognition.FaceTracker import FaceTracker
from ID_detection.yolov11.ID_Detection import detect_id_card
from Detection.Detection.settings import STATIC_ROOT
import time

IMAGE_FOLDER_PATH = os.path.join(STATIC_ROOT, 'images')

app = FastAPI()

# MongoDB connection setup
try:
    client = MongoClient(os.getenv("MONGO_URI", 'mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/'))
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
# camera_data = [{'camera_ip': 0,'camera_location':'lh_32'}]  # Adjust for your actual camera IP or path
camera_data = load_data()
print(camera_data)
# A dictionary to store frames for each camera
current_frames = {}
camera_trackers = {}# Function to capture frames from each camera
def capture_frame(camera_index, camera_ip):
    global camera_trackers
    if camera_index not in camera_trackers:
        camera_trackers[camera_index] = FaceTracker()

    face_tracker = camera_trackers[camera_index]
    cap = cv2.VideoCapture(camera_ip)  # RTSP stream URL

    if not cap.isOpened():
        print(f"Failed to open camera {camera_index} at {camera_ip}")
        return
    last_save_time = time.time()
    save_interval = 2.0  # Save interval in seconds
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            if frame_count % 1 == 0:
                # Process the frame for face and ID detection
                modified_frame, person_boxes, associations = detect_id_card(frame)
                modified_frame, flags  = recognize_faces_in_persons(modified_frame, person_boxes,face_tracker)
                # print(camera_data, camera_id)
                if time.time() - last_save_time > save_interval:
                    location = camera_data[camera_index]['camera_location']
                    # Save detections to MongoDB based on conditions
                    process_and_save_detections(
                        frame=frame,
                        person_bboxes=person_boxes,
                        flags=flags,
                        associations=associations,
                        camera_location=location
                    )

            # Encode the frame to JPEG and then base64
            _, jpeg = cv2.imencode('.jpg', frame)
            current_frames[camera_index] = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        else:
            print(f"Error capturing frame from camera {camera_index}")

# Start a separate thread for each camera to capture frames
for index, camera in enumerate(camera_data):
    threading.Thread(target=capture_frame, args=(index, camera["camera_ip"]), daemon=True).start()

# Function to process and save detections to MongoDB
def process_and_save_detections(frame, person_bboxes, flags, associations, camera_location):
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
        if (flag == "UNKNOWN") or (not wearing_id_card):
            image_name = f"person_{camera_location}_{current_time}_{idx}.jpg"
            image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)

            try:
                os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)  # Ensure directory exists
                cv2.imwrite(image_path, person_image)

                document = {
                    "_id": ObjectId(),
                    "location": camera_location,
                    "time": datetime.now().strftime("%D %I:%M %p") ,
                    "Role": "Unidentified" if flag == "UNKNOWN" else "Insider",
                    "Wearing_id_card": wearing_id_card,
                    "image": "/images/" + image_name,
                }

                result = collection.insert_one(document)
                print(f"Document inserted with _id: {result.inserted_id}")

            except Exception as e:
                print(f"Error saving detection data to database: {e}")

# FastAPI WebSocket endpoint for video feed
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

