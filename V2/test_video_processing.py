import cv2
import json
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from Backend.Person_detection.Person_detection import track_persons
from Backend.Face_recognition.face_recognize_lcnn import process_faces
from Backend.ID_detection.yolov11.ID_Detection import detect_id_card

# Load camera data from data.json
DATA_FILE_PATH = './Detection/data.json'

def load_camera_data():
    """Load camera configuration from JSON file"""
    if os.path.exists(DATA_FILE_PATH):
        try:
            with open(DATA_FILE_PATH, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")
            return []
    else:
        print(f"Error: File {DATA_FILE_PATH} does not exist.")
        return []

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

def process_camera_stream(camera_index, camera_ip, camera_location):
    """Process video from a single camera with AI analysis."""
    try:
        # Create a named window for this camera
        window_name = f"Camera {camera_index}: {camera_location}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        cap = cv2.VideoCapture(camera_ip)
        if not cap.isOpened():
            print(f"Failed to open camera {camera_index} at {camera_ip}")
            return

        frame_count = 0
        process_every_n_frames = 10  # Process every 10th frame
        start_time = datetime.now()
        fps = 0

        while True:
            # Calculate FPS every 30 frames
            if frame_count % 30 == 0:
                current_time = datetime.now()
                elapsed = (current_time - start_time).total_seconds()
                if elapsed > 0:
                    fps = 30 / elapsed
                start_time = current_time
            
            try:
                ret, frame = cap.read()
                if not ret:
                    print(f"Error capturing frame from camera {camera_index}")
                    cap.release()
                    cap = cv2.VideoCapture(camera_ip)
                    continue

                # Create a copy for annotations
                annotated_frame = frame.copy()
                frame_count += 1
                
                # Add FPS info to the frame
                cv2.putText(annotated_frame, f"Camera {camera_index}: {camera_location}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Process every Nth frame
                if frame_count % process_every_n_frames == 0:
                    # Show processing indicator
                    processing_frame = annotated_frame.copy()
                    cv2.putText(processing_frame, "Processing...", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(window_name, processing_frame)
                    cv2.waitKey(1)  # Update display
                    
                    # Person detection
                    person_results = track_persons(frame)

                    if person_results and "person_boxes" in person_results and "track_ids" in person_results:
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
                                    'face_detected': False,
                                    'face_box': [0, 0, 0, 0],
                                    'id_flag': False,
                                    'id_card': 'none',
                                    'id_box': [0, 0, 0, 0],
                                    'camera_location': camera_location,
                                }
                                
                                # Apply preprocessing to improve recognition
                                person_image = preprocess_frame(person_image)

                                # Face recognition
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

                                # Add person to the data list
                                people_data.append(person)

                            except Exception as e:
                                print(f"Error processing person: {e}")
                                continue

                        # Draw annotations on the frame
                        annotated_frame = draw_annotations(annotated_frame, people_data)

                # Display the annotated frame
                cv2.imshow(window_name, annotated_frame)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(f"Error in main processing loop: {e}")
                continue

    except Exception as e:
        print(f"Fatal error in process_camera_stream: {e}")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyWindow(window_name)

def main():
    # Load camera data
    camera_data = load_camera_data()
    print(f"Loaded {len(camera_data)} camera configurations")
    
    if not camera_data:
        print("No cameras found in configuration. Exiting.")
        return

    # Create a thread pool for processing multiple cameras
    with ThreadPoolExecutor(max_workers=len(camera_data)) as executor:
        # Start separate threads for each camera
        futures = []
        for index, camera in enumerate(camera_data):
            print(f"Starting camera {index}: {camera['camera_location']} ({camera['camera_ip']})")
            futures.append(executor.submit(
                process_camera_stream, 
                index, 
                camera["camera_ip"], 
                camera["camera_location"]
            ))
        
        # Wait for any key to be pressed in any window to exit
        print("Press 'q' in any window to exit")
        
        # Wait for all threads to complete
        for future in futures:
            future.result()  # This will block until completion

    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    main()