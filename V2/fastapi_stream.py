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

# Test MongoDB connection
try:
    db.command('ping')
    print("✅ MongoDB connection successful")
    print(f"Using database: {db.name}, Collection: {collection.name}")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")


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
    
    # Rest of your drawing code...
    try:
        print(f"Drawing annotations for {len(person_data)} people")
        for person in person_data:
            # Draw person bounding box
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw face bounding box if detected
            if person['face_detected']:
                fb_x1, fb_y1, fb_x2, fb_y2 = person['face_box']
                cv2.rectangle(frame, (fb_x1, fb_y1), (fb_x2, fb_y2), (0, 0, 255), 2)

            # Draw ID card box if detected
            if person['id_detected']:
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
# def get_adaptive_skip_frames(person_count, motion_score=0):
#     """Determine how many frames to skip based on scene complexity"""
#     # More people = process more frequently
#     if person_count > 3 or motion_score > 20:
#         return 2  # Process every 2 frames when busy
#     elif person_count > 1 or motion_score > 10:
#         return 5  # Process every 5 frames when moderate activity
#     return 1  # Process every 8 frames when scene is simple
def process_frame(camera_index, camera_ip, camera_location=""):
    try:
        cap = cv2.VideoCapture(camera_ip)

        # Set capture properties to maximum resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Verify the actual capture size
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera {camera_index} actual capture resolution: {actual_width}x{actual_height}")

        if not cap.isOpened():
            print(f"Failed to open camera {camera_index} at {camera_ip}")
            return

        frame_count = 0
        process_every_n_frames = 5  # Start with a moderate processing rate
        last_frame_time = datetime.now()

        while True:
            try:
                ret, original_frame = cap.read()  # Renamed to be explicit
                if not ret:
                    print(f"Error capturing frame from camera {camera_index}")
                    cap.release()
                    cap = cv2.VideoCapture(camera_ip)
                    continue

                frame_count += 1
                # Create a copy for annotations only
                annotated_frame = original_frame.copy()
                
                # Process every Nth frame
                if frame_count % process_every_n_frames == 0:
                    # Person detection on original frame
                    # The detection model will handle resize internally but return coordinates for original dimensions
                    person_results = track_persons(original_frame)

                    if not person_results or "person_boxes" not in person_results or "track_ids" not in person_results:
                        # Send the unprocessed frame since no detections
                        _, jpeg = cv2.imencode('.jpg', original_frame)
                        current_frames[camera_index] = jpeg
                        continue

                    person_boxes = person_results["person_boxes"]
                    track_ids = person_results["track_ids"]
                    people_data = []

                    for i, (person_box, track_id) in enumerate(zip(np.array(person_boxes).tolist(), track_ids)):
                        try:
                            x1, y1, x2, y2 = [int(coord) for coord in person_box]

                            # Validate and clip bounding boxes against the original frame
                            frame_height, frame_width, _ = original_frame.shape
                            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

                            # Crop the person from the original high-resolution frame
                            high_res_person_image = original_frame[y1:y2, x1:x2]
                            if high_res_person_image.size == 0:
                                print(f"Empty image for track_id: {track_id}")
                                continue

                            person = {
                                'bbox': [x1, y1, x2, y2],
                                'track_id': track_id,
                                'face_flag': "UNKNOWN",
                                'face_detected': False,
                                'face_box': [0, 0, 0, 0],
                                'face_confidence': 0.0,
                                'id_detected': False,
                                'id_card': 'none',
                                'id_box': [0, 0, 0, 0],
                                'camera_location': camera_location,
                            }
                            
                            # Apply preprocessing while maintaining resolution
                            enhanced_person_image = preprocess_frame(high_res_person_image)

                            # Face recognition - use track_id for temporal consistency
                            try:
                                print(f"About to call process_faces for track_id {track_id}, image shape: {enhanced_person_image.shape}")
                                person_flag, face_score, face_boxes = process_faces(enhanced_person_image, track_id=track_id)
                                print(f"Face processing results: flag={person_flag}, score={face_score}, boxes={face_boxes}")
                                
                                if face_boxes and len(face_boxes) > 0 and len(face_boxes[0]) == 4:
                                    fb_x1, fb_y1, fb_x2, fb_y2 = face_boxes[0]
                                    # Map face coordinates to original frame coordinates
                                    fb_x1 += x1
                                    fb_y1 += y1
                                    fb_x2 += x1
                                    fb_y2 += y1
                                    person['face_box'] = [fb_x1, fb_y1, fb_x2, fb_y2]
                                    person['face_flag'] = person_flag
                                    person['face_confidence'] = face_score
                                    person['face_detected'] = True
                                else:
                                    person['face_flag'] = "UNKNOWN"
                                    person['face_detected'] = False
                                    print("No face boxes returned by process_faces")
                            except Exception as e:
                                person['face_detected'] = False
                                print(f"Face recognition error: {e}")
                                import traceback
                                traceback.print_exc()

                            # ID card detection on the high-resolution crop
                            try:
                                id_detected, id_box, id_card = detect_id_card(enhanced_person_image)
                                person['id_detected'] = id_detected
                                person['id_card'] = id_card
                                
                                if id_detected and id_box:
                                    ib_x1, ib_y1, ib_x2, ib_y2 = id_box
                                    # Map ID box coordinates to original frame coordinates
                                    ib_x1 += x1
                                    ib_y1 += y1
                                    ib_x2 += x1
                                    ib_y2 += y1
                                    person['id_box'] = [ib_x1, ib_y1, ib_x2, ib_y2]
                            except Exception as e:
                                print(f"ID card detection error: {e}")

                            print(f"Person {i}: face_detected={person['face_detected']}, face_flag={person['face_flag']}, id_detected={person['id_detected']}")

                            # Add person to the list if face detected or ID card detected
                            if person['face_detected']:
                                people_data.append(person)
                            
                            # Save information to MongoDB if needed
                            print("Person detect & id flag :", person['face_detected'], person['id_detected'])
                            if person['face_detected']:  # First ensure face is detected
                                should_save = False
                                save_method = None
                                
                                # Condition 1: Unknown face (face_flag="UNKNOWN")
                                if person['face_flag'] == "UNKNOWN" and frame_count % 30 == 0:
                                    should_save = True
                                    save_method = "data_manager"  # Use the DataManager for UNKNOWN faces
                                
                                # Condition 2: No ID card
                                if not person['id_detected'] and frame_count % 10 == 0:
                                    should_save = True
                                    save_method = "direct"  # Use direct MongoDB insert for no-ID cases
                                
                                # Save using the appropriate method
                                if should_save:
                                    try:
                                        if save_method == "data_manager":
                                            # Save the high-resolution image
                                            saved_doc = data_manager.save_data(enhanced_person_image, person)
                                            print(f"✅ Saved unknown face via DataManager: {saved_doc}")
                                        else:
                                            # Use your existing direct MongoDB insert code with high-res image
                                            image_name = f"{camera_index}-{camera_location}-{track_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg"
                                            doc_id = ObjectId()
                                            document = {
                                                "_id": doc_id,
                                                "timestamp": datetime.now(),
                                                "person_id": None,
                                                "camera_location": camera_location,
                                                "id_detected": person['id_detected'],
                                                'bbox': person['bbox'],
                                                'track_id': person['track_id'],
                                                'face_detected': person['face_detected'],
                                                'face_confidence': person['face_confidence'],
                                                'face_box': person['face_box'],
                                                'id_card': person['id_card'],
                                                'id_box': person['id_box'],
                                                'image_path': 'images/'+image_name
                                            }
                                            collection.insert_one(document)
                                            path = os.path.join(IMAGE_FOLDER_PATH, image_name)
                                            cv2.imwrite(path, enhanced_person_image)
                                            print(f"✅ Saved no-ID person via direct MongoDB: {doc_id}")
                                    except Exception as e:
                                        print(f"❌ Error saving to DB: {str(e)}")

                        except Exception as e:
                            print(f"Error processing person: {e}")
                            continue

                    # Draw annotations on a copy of the frame
                    annotated_frame = draw_annotations(annotated_frame, people_data)

                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Update current frame for websocket (use annotated frame)
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



