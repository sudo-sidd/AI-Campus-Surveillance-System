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

app = FastAPI()


# Load camera data from data.json
DATA_FILE_PATH = Path('../Detection/data.json')
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

        while True:
            try:
                ret, frame = cap.read()
                if ret:
                    # Update current frame for websocket
                    frame = cv2.flip(frame, 1)
                    _, jpeg = cv2.imencode('.jpg', frame)
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