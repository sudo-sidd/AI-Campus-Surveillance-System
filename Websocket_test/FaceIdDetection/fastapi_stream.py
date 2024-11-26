import base64
import cv2
from fastapi import FastAPI, WebSocket
import threading
import json
import asyncio
from pathlib import Path
import os

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
# camera_data = load_data()
camera_data = [{'camera_ip':0}]

# A dictionary to store frames for each camera
current_frames = {}

def capture_frame(camera_index, camera_ip):
    cap = cv2.VideoCapture(camera_ip)  # RTSP stream URL
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index} at {camera_ip}")
        return
    
    while True:
        ret, frame = cap.read()
        if ret:
            # Encode the frame to JPEG and then base64
            _, jpeg = cv2.imencode('.jpg', frame)
            current_frames[camera_index] = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        else:
            print(f"Error capturing frame from camera {camera_index}")


# Start a separate thread for each camera to capture frames
for index, camera in enumerate(camera_data):
    threading.Thread(target=capture_frame, args=(index, camera["camera_ip"]), daemon=True).start()

@app.websocket("/ws/video/{camera_id}/")
async def video_feed(websocket: WebSocket, camera_id: int):
    await websocket.accept()
    
    while True:
        if camera_id in current_frames:
            frame = current_frames[camera_id]
            # Send the latest frame to the client as a base64-encoded string
            await websocket.send_text(f'{{"frame": "{frame}"}}')
        await asyncio.sleep(0.1)  # Sleep to prevent excessive CPU usage


