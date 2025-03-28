import base64
import cv2
import json
import os
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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
import time
from fastapi.middleware.cors import CORSMiddleware
import torch



IMAGE_FOLDER_PATH = os.path.join(STATIC_ROOT, 'images')
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://0.0.0.0:8080", "http://192.168.8.86:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
print(f"Loaded {len(camera_data)} camera configurations")

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
            # Add confidence percentage for better visibility
            confidence_text = f"{person['face_confidence']*100:.1f}%" if 'face_confidence' in person else ""
            text = f"ID: {person['track_id']} | Face: {person['face_flag']} {confidence_text} | IDCard: {person['id_card']}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Add background and text
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (255, 255, 255), -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Add confidence bar visualization if confidence is available
            if 'face_confidence' in person and person['face_confidence'] > 0:
                bar_length = 50
                filled_length = int(bar_length * person['face_confidence'])
                y_pos = y1 - 30
                
                # Draw empty bar
                cv2.rectangle(frame, (x1, y_pos), (x1 + bar_length, y_pos + 5), (0, 0, 0), 1)
                
                # Fill with color based on confidence
                if person['face_confidence'] < 0.4:
                    color = (0, 0, 255)  # Red
                elif person['face_confidence'] < 0.6:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Green
                    
                cv2.rectangle(frame, (x1, y_pos), (x1 + filled_length, y_pos + 5), color, -1)

    except Exception as e:
        print(f"Error in draw_annotations: {e}")
    return frame

<<<<<<< HEAD
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

=======
>>>>>>> 43b92e29e69eb3e299f97a7f264304190a3020b5
def preprocess_frame(frame):
    """Enhances contrast using CLAHE in LAB color space."""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
<<<<<<< HEAD

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)

        # Merge channels and convert back to BGR
        merged_lab = cv2.merge((l_clahe, a_channel, b_channel))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return frame

# def preprocess_frame(frame):
#     """Enhances contrast using CLAHE in LAB color space."""
#     try:
#         # Convert to LAB color space
#         lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#         l_channel, a_channel, b_channel = cv2.split(lab)
=======
>>>>>>> 43b92e29e69eb3e299f97a7f264304190a3020b5

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)

        # Merge channels and convert back to BGR
        merged_lab = cv2.merge((l_clahe, a_channel, b_channel))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return frame

def get_adaptive_skip_frames(person_count, motion_score=0):
    """Determine how many frames to skip based on scene complexity"""
    # More people = process more frequently
    if person_count > 3 or motion_score > 20:
        return 2  # Process every 2 frames when busy
    elif person_count > 1 or motion_score > 10:
        return 5  # Process every 5 frames when moderate activity
<<<<<<< HEAD
    return 1  # Process every 8 frames when scene is simple
=======
    return 8  # Process every 8 frames when scene is simple
>>>>>>> 43b92e29e69eb3e299f97a7f264304190a3020b5

def process_frame(camera_index, camera_ip, camera_location=""):
    try:
        cap = cv2.VideoCapture(camera_ip)

        if not cap.isOpened():
            print(f"Failed to open camera {camera_index} at {camera_ip}")
            return

        frame_count = 0
        process_every_n_frames = 5  # Start with a moderate processing rate
        prev_frame = None
        last_frame_time = datetime.now()

        while True:
            try:
                # Calculate fps occasionally
                if frame_count % 30 == 0:
                    now = datetime.now()
                    elapsed = (now - last_frame_time).total_seconds()
                    if elapsed > 0:
                        fps = 30 / elapsed
                        print(f"Camera {camera_index}: {fps:.1f} FPS | Processing every {process_every_n_frames} frames")
                    last_frame_time = now
                
                ret, frame = cap.read()
                if not ret:
                    print(f"Error capturing frame from camera {camera_index}")
                    cap.release()
                    cap = cv2.VideoCapture(camera_ip)
                    continue

                frame_count += 1
                annotated_frame = frame.copy()
                
                # Calculate frame motion if we have a previous frame
                motion_score = 0
                if prev_frame is not None:
                    frame_diff = cv2.absdiff(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    )
                    motion_score = np.mean(frame_diff)

                # Process every Nth frame
                if frame_count % process_every_n_frames == 0:
                    # Store this frame for motion detection
                    prev_frame = frame.copy()
                    
                    # Person detection
                    person_results = track_persons(frame)

                    if not person_results or "person_boxes" not in person_results or "track_ids" not in person_results:
                        # Send the unprocessed frame since no detections
                        _, jpeg = cv2.imencode('.jpg', frame)
                        current_frames[camera_index] = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                        continue

                    person_boxes = person_results["person_boxes"]
                    track_ids = person_results["track_ids"]
                    people_data = []
                    
                    # Adjust processing rate based on number of people
                    person_count = len(person_boxes)
                    process_every_n_frames = get_adaptive_skip_frames(person_count, motion_score)

                    for person_box, track_id in zip(np.array(person_boxes).tolist(), track_ids):
                        try:
                            x1, y1, x2, y2 = [int(coord) for coord in person_box]

                            # Validate and clip bounding boxes
                            frame_height, frame_width, _ = frame.shape
                            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

<<<<<<< HEAD
                            # Crop the person image
                            person_image = frame[y1:y2, x1:x2]
                            if person_image.size == 0:
                                print(f"Empty image for track_id: {track_id}")
                                continue

=======
                            # Skip if box is too small
                            if (x2 - x1) < 40 or (y2 - y1) < 80:
                                continue

                            # Crop the person image
                            person_image = frame[y1:y2, x1:x2]
                            if person_image.size == 0:
                                print(f"Empty image for track_id: {track_id}")
                                continue

>>>>>>> 43b92e29e69eb3e299f97a7f264304190a3020b5
                            person = {
                                'bbox': [x1, y1, x2, y2],
                                'track_id': track_id,
                                'face_flag': "UNKNOWN",
                                'face_detected': False,
                                'face_box': [0, 0, 0, 0],
                                'face_confidence': 0.0,
                                'id_flag': False,
                                'id_card': 'none',
                                'id_box': [0, 0, 0, 0],
                                'camera_location': camera_location,
                            }
                            
                            # Apply preprocessing to improve recognition
                            person_image = preprocess_frame(person_image)

                            # Face recognition - use track_id for temporal consistency
                            try:
                                person_flag, face_score, face_boxes = process_faces(person_image, track_id=track_id)
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
                                    person['face_confidence'] = face_score
                                else:
                                    person['face_flag'] = "UNKNOWN"
                                    person['face_detected'] = False
                            except Exception as e:
                                print(f"Face recognition error: {e}")

                            # ID card detection
                            try:
                                id_flag, id_box, id_card = detect_id_card(person_image)
                                person['id_flag'] = id_flag
                                person['id_card'] = id_card
                                
                                if id_flag and id_box:
                                    ib_x1, ib_y1, ib_x2, ib_y2 = id_box
                                    # Adjust coordinates to original frame
                                    ib_x1 += x1
                                    ib_y1 += y1
                                    ib_x2 += x1
                                    ib_y2 += y1
                                    person['id_box'] = [ib_x1, ib_y1, ib_x2, ib_y2]
                            except Exception as e:
                                print(f"ID card detection error: {e}")

                            # Add person to the list if face detected or ID card detected
                            if person['face_detected'] or person['id_flag']:
                                people_data.append(person)
                            
                            # Save information to MongoDB if needed
                            # Store unknown faces or if no ID card
<<<<<<< HEAD
                            print("Person detect & id flag :",person['face_detected'] , person['id_flag'])
=======
>>>>>>> 43b92e29e69eb3e299f97a7f264304190a3020b5
                            if person['face_detected'] or person['id_flag']:
                                if id_flag == False:
                                    # Only save to database periodically to avoid flooding database
                                    # with duplicate entries
                                    if frame_count % 20 == 0:  # Save every 20th time we process
                                        image_name = f"{camera_index}-{camera_location}-{track_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg"
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
                                            'face_confidence': person['face_confidence'],
                                            'face_box': person['face_box'],
                                            'id_card': person['id_card'],
                                            'id_box': person['id_box'],
                                            'image_path': 'images/'+image_name
                                        }
                                        collection.insert_one(document)
                                        path = os.path.join(IMAGE_FOLDER_PATH, image_name)
                                        cv2.imwrite(path, person_image)
                                
                                # Only save unknown faces every 30th frame to avoid duplicates
                                if person['face_flag'] == "UNKNOWN" and frame_count % 30 == 0:
                                    saved_doc = data_manager.save_data(person_image, person)
                                    print(f"Saved unknown person: {saved_doc}")

                        except Exception as e:
                            print(f"Error processing person: {e}")
                            continue

                    # Draw annotations on a copy of the frame
                    annotated_frame = draw_annotations(annotated_frame, people_data)
<<<<<<< HEAD

                # cv2.imshow("frame", annotated_frame)
            

                # # Exit if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
=======
>>>>>>> 43b92e29e69eb3e299f97a7f264304190a3020b5

                _, jpeg = cv2.imencode('.jpg', annotated_frame)
                current_frames[camera_index] = jpeg # ⚠️ Don't modify this code
                
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
    print(f"Starting camera {index}: {camera['camera_location']} ({camera['camera_ip']})")
    executor.submit(process_frame, index, camera["camera_ip"], camera["camera_location"])


# ⚠️Don't modify the below code!!
# FastAPI with a standard streaming endpoint
@app.get("/video/{camera_id}")
async def video_feed(camera_id: int):
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

def generate_frames(camera_id):
    """Generator function that yields video frames for streaming."""
    try:
        while True:
            if camera_id in current_frames:
                # Decode the base64 frame back to bytes
                frame_bytes = current_frames[camera_id]
            
                # Yield the frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes.tobytes() + b'\r\n')
            
            # Small delay to control frame rate
            # time.sleep(0.1)
    except Exception as e:
        print(f"Error in generate_frames: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)



