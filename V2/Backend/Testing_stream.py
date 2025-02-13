import cv2
import numpy as np
from Person_detection.Person_detection_test import track_persons
from Face_recognition.face_recognize_lcnn import process_faces
from ID_detection.yolov11.ID_Detection_test import detect_id_card

# Dictionary to store face recognition history for each track ID
face_recognition_memory = {}
ema_alpha = 0.2  # Exponential Moving Average factor (higher = faster updates)
frame_memory = 15  # Number of frames to wait before marking as UNKNOWN

def draw_annotations(frame, person_data):
    """Draw bounding boxes and annotations on the frame."""
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


def update_recognition_memory(track_id, new_name, new_score):
    """Update recognition memory with a rolling average confidence score."""
    if track_id not in face_recognition_memory:
        face_recognition_memory[track_id] = {
            "name": new_name,
            "score": new_score,
            "low_conf_frames": 0  # Counter for low-confidence frames
        }
        return new_name  # First-time recognition

    memory = face_recognition_memory[track_id]

    # Exponential Moving Average (EMA) for confidence score smoothing
    memory["score"] = (ema_alpha * new_score) + ((1 - ema_alpha) * memory["score"])

    # If new recognition is UNKNOWN but memory score is still high, keep previous identity
    if new_name == "UNKNOWN":
        memory["low_conf_frames"] += 1  # Increase low confidence counter

        # If low confidence persists for too long, reset to UNKNOWN
        if memory["low_conf_frames"] > frame_memory:
            memory["name"] = "UNKNOWN"
    else:
        # Reset low confidence counter if a valid name is detected
        memory["name"] = new_name
        memory["low_conf_frames"] = 0

    return memory["name"]

#rtsp://aiml:Siet@2727@192.168.3.183:554/Streaming/Channels/101
cap = cv2.VideoCapture("rtsp://aiml:Siet@2727@192.168.3.183:554/Streaming/Channels/101")

if not cap.isOpened():
    print(f"Failed to open camera")
    exit()

frame_count = 0
process_every_n_frames = 5

while True:
    ret, frame = cap.read()
    if ret:
        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            try:
                # Detect and track persons
                person_results = track_persons(frame)

                # Check if person_results contains valid data
                if not person_results or "modified_frame" not in person_results or "person_boxes" not in person_results:
                    print("Invalid person results")
                    continue

                # frame = person_results["modified_frame"]
                person_boxes = person_results["person_boxes"]
                track_ids = person_results["track_ids"]
                people_data = []

                for person_box, track_id in zip(np.array(person_boxes).tolist(), track_ids):
                    x1, y1, x2, y2 = [int(coord) for coord in person_box]

                    # Validate and clip bounding boxes
                    frame_height, frame_width, _ = frame.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

                    # Crop the person image
                    person_image = frame[y1:y2, x1:x2]
                    if person_image.size == 0:
                        print(f"Empty image for track_id: {track_id}")
                        continue

                    # Initialize person data
                    person = {
                        'bbox': [x1, y1, x2, y2],
                        'track_id': track_id,
                        'face_flag': "UNKNOWN",
                        'face_box': [0, 0, 0, 0],
                        'id_flag': False,
                        'id_card': 'none',
                        'id_box': [0, 0, 0, 0]
                    }
                    person_image = preprocess_frame(person_image)
                    # Face recognition
                    try:
                        person_flag, face_score, face_boxes = process_faces(person_image)
                        if face_boxes:
                            fb_x1, fb_y1, fb_x2, fb_y2 = face_boxes[0]
                            # Adjust coordinates to original frame
                            fb_x1 += x1
                            fb_y1 += y1
                            fb_x2 += x1
                            fb_y2 += y1

                            person['face_flag'] = update_recognition_memory(track_id, person_flag, face_score)
                            person['face_score'] = face_score
                            person['face_box'] = [fb_x1, fb_y1, fb_x2, fb_y2]
                        else:
                            person['face_flag'] = "UNKNOWN"
                            person['face_score'] = 0.0
                    except Exception as e:
                        print(f"Face recognition error: {e}")
                        person['face_flag'] = "UNKNOWN"
                        person['face_score'] = 0.0

                    try:
                        id_flag, id_box, id_card = detect_id_card(person_image)
                        person['id_flag'] = id_flag
                        person['id_box'] = id_box
                        person['id_card'] = id_card

                    except Exception as e:
                        print(f"ID card detection error: {e}")

                    if person['face_detected'] == True:
                        people_data.append(person)

                name_scores = {}
                for person in people_data:
                    name = person['face_flag']
                    if name != "UNKNOWN":
                        current_score = person['face_score']
                        if name not in name_scores or current_score > name_scores[name]['score']:
                            name_scores[name] = {'score': current_score, 'track_id': person['track_id']}

                # Update people_data to mark duplicates as UNKNOWN
                for person in people_data:
                    name = person['face_flag']
                    if name != "UNKNOWN":
                        max_info = name_scores.get(name)
                        if max_info:
                            if (person['face_score'] < max_info['score']) or \
                                    (person['face_score'] == max_info['score'] and person['track_id'] != max_info[
                                        'track_id']):
                                person['face_flag'] = "UNKNOWN"
                                person['face_box'] = None


                # Draw bounding boxes and annotations
                draw_annotations(frame, people_data)

                print("people data:", people_data)

            except Exception as e:
                print(f"Error during frame processing: {e}")

            img = cv2.resize(frame, (640*2, 480*2))
            cv2.imshow('frame', img)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()