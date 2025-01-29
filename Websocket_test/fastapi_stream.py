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
from SaveData.SaveData import DataManager

IMAGE_FOLDER_PATH = os.path.join(STATIC_ROOT, 'images')
app = FastAPI()

data_manager = DataManager(
        mongo_uri=os.getenv("MONGO_URI", "mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/"),
        db_name='ML_project',
        collection_name='DatabaseDB'
    )

# Load camera data from data.json
DATA_FILE_PATH = Path('/home/mithun/PROJECT/git_update/Face_rec-ID_detection/Websocket_test/Detection/data.json')
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
    try:
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
    except Exception as e:
        print(f"Error in draw_annotations: {e}")
    return frame

def save_person_image(frame, bbox, camera_location, track_id):
    try:
        x1, y1, x2, y2 = bbox
        person_image = frame[y1:y2, x1:x2]
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"person_{camera_location}_{track_id}_{current_time}.jpg"
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)
        
        os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)
        cv2.imwrite(image_path, person_image)
        return "/images/" + image_name
    except Exception as e:
        print(f"Error saving person image: {e}")
        return None

def process_frame(camera_index, camera_ip, camera_location):
    try:
        cap = cv2.VideoCapture(camera_ip)

        if not cap.isOpened():
            print(f"Failed to open camera {camera_index} at {camera_ip}")
            return

        frame_count = 0

        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print(f"Error capturing frame from camera {camera_index}")
                    cap.release()
                    cap = cv2.VideoCapture(camera_ip)
                    continue

                frame_count += 1

                if frame_count % 3 == 0:
                    person_results = track_persons(frame)

                    if not person_results or "person_boxes" not in person_results or "track_ids" not in person_results:
                        continue

                    person_boxes = person_results["person_boxes"]
                    track_ids = person_results["track_ids"]
                    people_data = []

                    for person_box, track_id in zip(np.array(person_boxes).tolist(), track_ids):
                        try:
                            x1, y1, x2, y2 = [int(coord) for coord in person_box]
                            frame_height, frame_width = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame_width, x2), min(frame_height, y2)

                            if x2 <= x1 or y2 <= y1:
                                continue

                            person_image = frame[y1:y2, x1:x2]
                            if person_image.size == 0:
                                continue

                            person = {
                                'bbox': [x1, y1, x2, y2],
                                'track_id': track_id,
                                'face_flag': "UNKNOWN",
                                'face_box': [0, 0, 0, 0],
                                'id_flag': False,
                                'id_card': 'none',
                                'id_box': [0, 0, 0, 0],
                                'camera_location':camera_location,
                            }

                            try:
                                person_flag, face_box = process_faces(person_image)
                                person['face_flag'] = person_flag
                                person['face_box'] = face_box
                            except Exception as e:
                                print(f"Face recognition error: {e}")

                            try:
                                id_flag, id_box, id_card = detect_id_card(person_image)
                                person['id_flag'] = id_flag
                                person['id_box'] = id_box
                                person['id_card'] = id_card
                            except Exception as e:
                                print(f"ID card detection error: {e}")


                            people_data.append(person)
                            if person_flag == "UNKNOWN" or id_flag == False:
                                saved_doc = data_manager.save_data(person_image, person)

                        except Exception as e:
                            print(f"Error processing person: {e}")
                            continue

                    # Draw annotations on a copy of the frame
                    annotated_frame = frame.copy()
                    annotated_frame = draw_annotations(annotated_frame, people_data)


                    # Update current frame for websocket
                    _, jpeg = cv2.imencode('.jpg', annotated_frame)
                    current_frames[camera_index] = base64.b64encode(jpeg.tobytes()).decode('utf-8')

            except Exception as e:
                print(f"Error in main processing loop: {e}")
                continue

    except Exception as e:
        print(f"Fatal error in process_frame: {e}")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()

# Start separate threads for each camera
for index, camera in enumerate(camera_data):
    executor.submit(process_frame, index, camera["camera_ip"],camera["camera_location"])

@app.websocket("/ws/video/{camera_id}/")
async def video_feed(websocket: WebSocket, camera_id: int):
    await websocket.accept()
    try:
        while True:
            if camera_id in current_frames:
                await websocket.send_text(f'{{"frame": "{current_frames[camera_id]}"}}')
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)