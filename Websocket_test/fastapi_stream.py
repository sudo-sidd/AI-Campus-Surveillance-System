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

mongo_uri=os.getenv("MONGO_URI", "mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/")
db_name='ML_project'
collection_name='DatabaseDB'

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]


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
            text = f"ID: {track_id} | Face: {face_flag[0]} | IDCard: {id_card}"
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

        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print(f"Error capturing frame from camera {camera_index}")
                    cap.release()
                    cap = cv2.VideoCapture(camera_ip)
                    continue
                frame = preprocess_frame(frame)
                frame_count += 1

                if frame_count % 10 == 0:
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
                                'camera_location': camera_location,
                            }

                            try:
                                person_flag, face_box = process_faces(person_image)
                                person['face_flag'] = person_flag[0]
                                person['face_box'] = face_box
                            except Exception as e:
                                print(f"Face recognition error: {e}")

                            # CHEST-LEVEL ID DETECTION ENHANCEMENTS
                            height, width = person_image.shape[:2]

                            # 1. Adjusted ROI for chest-level IDs (20%-60% of person height)
                            roi_y_start = int(height * 0.20)
                            roi_y_end = int(height * 0.60)
                            roi_x_start = int(width * 0.25)
                            roi_x_end = int(width * 0.75)
                            id_roi = person_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

                            # 2. ROI Enhancement
                            if id_roi.size > 0:
                                # Sharpening and upscaling
                                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                                sharpened = cv2.filter2D(id_roi, -1, kernel)
                                enhanced_roi = cv2.resize(sharpened, None, fx=1.5, fy=1.5,
                                                          interpolation=cv2.INTER_CUBIC)

                                # 3. Rotation augmentation (±15 degrees)
                                angle = np.random.uniform(-15, 15)
                                rows, cols = enhanced_roi.shape[:2]
                                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                                rotated_roi = cv2.warpAffine(enhanced_roi, M, (cols, rows))
                            else:
                                rotated_roi = id_roi

                            try:
                                id_flag, id_box_roi, id_card = detect_id_card(rotated_roi)

                                # Convert coordinates back to original frame
                                if id_flag and id_box_roi:
                                    # Adjust for ROI position and enhancement/rotation
                                    rx1, ry1, rx2, ry2 = id_box_roi
                                    scale_factor = 1 / 1.5  # Account for upscaling

                                    # Original ROI coordinates
                                    orig_x1 = int(rx1 * scale_factor) + roi_x_start + x1
                                    orig_y1 = int(ry1 * scale_factor) + roi_y_start + y1
                                    orig_x2 = int(rx2 * scale_factor) + roi_x_start + x1
                                    orig_y2 = int(ry2 * scale_factor) + roi_y_start + y1

                                    person['id_box'] = [orig_x1, orig_y1, orig_x2, orig_y2]
                                    person['id_card'] = id_card
                                else:
                                    person['id_box'] = [0, 0, 0, 0]

                                person['id_flag'] = id_flag
                            except Exception as e:
                                print(f"ID card detection error: {e}")

                            people_data.append(person)

                            # Save condition for chest-level IDs
                            if person['face_flag'] == "UNKNOWN" or not person['id_flag']:
                                save_doc = data_manager.save_document(
                                    person_image,
                                    {
                                        'camera_location': camera_location,
                                        'id_flag': person['id_flag'],
                                        'bbox': person['bbox'],
                                        'track_id': track_id,
                                        'face_flag': person['face_flag'],
                                        'face_box': person['face_box'],
                                        'id_card': person['id_card'],
                                        'id_box': person['id_box'],
                                    }
                                )
                                print(save_doc)

                        except Exception as e:
                            print(f"Error processing person: {e}")
                            continue

                    # Draw annotations with ID box
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