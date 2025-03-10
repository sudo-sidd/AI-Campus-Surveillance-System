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
from Backend.Person_detection.Person_detection import track_persons
from Backend.Face_recognition.face_recognize_lcnn import process_faces
from Backend.ID_detection.yolov11.ID_Detection import detect_id_card
from Backend.SaveData.SaveData import DataManager


IMAGE_FOLDER_PATH = os.path.join(STATIC_ROOT, 'images')
app = FastAPI()

data_manager = DataManager(
        mongo_uri=os.getenv("MONGO_URI", "mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/"),
        db_name='ML_project',
        collection_name='DatabaseDB'
    )

mongo_uri=os.getenv("MONGO_URI", "mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/")
db_name='ML_project'
collection_name='DatabaseDB'

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]


# Load camera data from data.json
DATA_FILE_PATH = './Detection/data.json'
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
            # Draw person bounding box
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw face bounding box if detected
            if person['face_detected']:
                fb_x1, fb_y1, fb_x2, fb_y2 = person['face_box']
                cv2.rectangle(frame, (fb_x1, fb_y1), (fb_x2, fb_y2), (0, 0, 255), 2)

            # Draw ID card box if detected
            if person['id_flag']:
                ib_x1, ib_y1, ib_x2, ib_y2 = person['id_box']
                cv2.rectangle(frame, (ib_x1, ib_y1), (ib_x2, ib_y2), (255, 0, 0), 2)

            # Prepare text annotations
            text = f"ID: {person['track_id']} | Face: {person['face_flag']} | IDCard: {person['id_card']}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Add background and text
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (255, 255, 255), -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    except Exception as e:
        print(f"Error in draw_annotations: {e}")
    return frame

def preprocess_frame(frame):
    """Enhances contrast using CLAHE in LAB color space."""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)

        # Merge channels and convert back to BGR
        merged_lab = cv2.merge((l_clahe, a_channel, b_channel))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return frame

def process_frame(camera_index, camera_ip, camera_location):
    try:
        cap = cv2.VideoCapture(camera_ip)

        if not cap.isOpened():
            print(f"Failed to open camera {camera_index} at {camera_ip}")
            return

        frame_count = 0
        process_every_n_frames = 10

        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print(f"Error capturing frame from camera {camera_index}")
                    cap.release()
                    cap = cv2.VideoCapture(camera_ip)
                    continue

                frame_count += 1

                if frame_count % process_every_n_frames == 0:
                    person_results = track_persons(frame)

                    if not person_results or "person_boxes" not in person_results or "track_ids" not in person_results:
                        continue

                    person_boxes = person_results["person_boxes"]
                    track_ids = person_results["track_ids"]
                    people_data = []

                    for person_box, track_id in zip(np.array(person_boxes).tolist(), track_ids):
                        try:
                            x1, y1, x2, y2 = [int(coord) for coord in person_box]

                            # Validate and clip bounding boxes
                            frame_height, frame_width, _ = frame.shape
                            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

                            # Crop the person image
                            person_image = frame[y1:y2, x1:x2]
                            if person_image.size == 0:
                                print(f"Empty image for track_id: {track_id}")
                                continue

                            person = {
                                'bbox': [x1, y1, x2, y2],
                                'track_id': track_id,
                                'face_flag': "UNKNOWN",
                                'face_detected': False,  # Add this field
                                'face_box': [0, 0, 0, 0],
                                'id_flag': False,
                                'id_card': 'none',
                                'id_box': [0, 0, 0, 0],
                                'camera_location':camera_location,
                            }
                            person_image = preprocess_frame(person_image)

                            #Face recognition

                            try:
                                person_flag, face_score, face_boxes = process_faces(person_image)
                                if face_boxes and len(face_boxes) > 0:
                                    fb_x1, fb_y1, fb_x2, fb_y2 = face_boxes[0]
                                    # Adjust coordinates to original frame
                                    fb_x1 += x1
                                    fb_y1 += y1
                                    fb_x2 += x1
                                    fb_y2 += y1
                                    person['face_box'] = [fb_x1, fb_y1, fb_x2, fb_y2]
                                    person['face_flag'] = person_flag
                                    person['face_detected'] = True
                                else:
                                    person['face_flag'] = "UNKNOWN"
                                    person['face_detected'] = False

                            except Exception as e:
                                print(f"Face recognition error: {e}")

                            try:
                                id_flag, id_box, id_card = detect_id_card(person_image)
                                person['id_flag'] = id_flag
                                person['id_box'] = id_box
                                person['id_card'] = id_card
                            except Exception as e:
                                print(f"ID card detection error: {e}")

                            if person['face_detected'] or person['id_flag']:
                                people_data.append(person)
                            # people_data.append(person)
                            
                            if id_flag == False :
                                image_name = f"{camera_index}-{camera_location}-{track_id}-{datetime.now()}.jpg"
                                doc_id = ObjectId()
                                document = {
                                    "_id": doc_id,
                                    "timestamp": datetime.now(),
                                    "person_id": None,  # Link to existing person,
                                    "camera_location": camera_location,
                                    "id_flag": person['id_flag'],
                                    'bbox': person['bbox'],
                                    'track_id': person['track_id'],
                                    'face_flag': person['face_flag'],
                                    'face_box': person['face_box'],
                                    'id_card': person['id_card'],
                                    'id_box': person['id_box'],
                                    'image_path': 'images/'+image_name
                                }
                                collection.insert_one(document)
                                path = os.path.join(IMAGE_FOLDER_PATH, image_name)
                                cv2.imwrite(path, person_image)
                            if person['face_flag'] == "UNKNOWN":
                                saved_doc = data_manager.save_data(person_image, person)
                                print(saved_doc)

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