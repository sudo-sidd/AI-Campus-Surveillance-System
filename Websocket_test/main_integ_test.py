from Face_recognition.face_recognize_yolo  import recognize_faces_in_persons
from ID_detection.yolov11.ID_Detection import detect_id_card
import cv2
import threading
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from queue import Queue
import io
import sys
import os
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId

# Initialize FastAPI app
app = FastAPI()

# Initialize a queue to hold frames
frame_queue = Queue(maxsize=10)


username = "aiml"
password = "Siet@2727"
camera_ip = "192.168.3.148"
port = "554"  # Default RTSP port for Hikvision cameras

# Construct the RTSP URL
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:{port}/Streaming/Channels/101"
# Use OpenCV to capture video from the webcam (camera index 0)
# MongoDB connection
try:
    client = MongoClient(os.getenv("MONGO_URI", 'mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/'))
    db = client['ML_project']  # Replace with your database name
    collection = db['DatabaseDB']  # Replace with your collection name
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    sys.exit(1)  # Exit if database connection fails

cap = cv2.VideoCapture(rtsp_url)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Couldn't access the webcam.")
    exit()
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


# Define a function to continuously capture frames from the webcam in the background
def capture_frames():

    last_save_time = time.time()
    save_interval = 2.0  # Save interval in seconds
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read frame from webcam.")
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
                    camera_id=rtsp_url
                )
                last_save_time = time.time()
        # Add the frame to the queue if there's space
        if not frame_queue.full():
            frame_queue.put(modified_frame)

        # Simulate processing time
        time.sleep(0.05)

    cap.release()

# Start a background thread to capture frames from the webcam
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

# Define a function to convert a frame to a byte stream (JPEG)
def frame_to_bytes(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    return img_encoded.tobytes()

# API route to stream the latest webcam frame
@app.get("/get_frame")
async def get_frame():
    if not frame_queue.empty():
        frame = frame_queue.get()
        frame_bytes = frame_to_bytes(frame)
        return StreamingResponse(io.BytesIO(frame_bytes), media_type="image/jpeg")
    else:
        return {"error": "No frames available yet."}

# Run the FastAPI server with Uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
