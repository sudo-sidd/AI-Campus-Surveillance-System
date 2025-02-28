import cv2
import numpy as np
from Person_detection.Person_detection_test import track_persons
from Face_recognition.face_recognize_lcnn import process_faces
from ID_detection.yolov11.ID_Detection_test import detect_id_card

# Dictionary to store face recognition history for each track ID
face_recognition_memory = {}
ema_alpha = 0.2  # Exponential Moving Average factor
frame_memory = 15  # Frames to wait before marking as UNKNOWN


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
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def preprocess_frame(frame):
    """Enhances contrast using CLAHE in LAB color space."""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)

        merged_lab = cv2.merge((l_clahe, a_channel, b_channel))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return frame


def update_recognition_memory(track_id, new_name, new_score):
    """Update recognition memory with a rolling average confidence score."""
    if track_id not in face_recognition_memory:
        face_recognition_memory[track_id] = {"name": new_name, "score": new_score, "low_conf_frames": 0}
        return new_name

    memory = face_recognition_memory[track_id]
    memory["score"] = (ema_alpha * new_score) + ((1 - ema_alpha) * memory["score"])

    if new_name == "UNKNOWN":
        memory["low_conf_frames"] += 1
        if memory["low_conf_frames"] > frame_memory:
            memory["name"] = "UNKNOWN"
    else:
        memory["name"] = new_name
        memory["low_conf_frames"] = 0

    return memory["name"]


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    exit()

frame_count = 0
process_every_n_frames = 5

while True:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        print(f"Frame Count: {frame_count}")

        if frame_count % process_every_n_frames == 0:
            try:
                person_results = track_persons(frame)
                print("Track Persons Output:", person_results)

                if not person_results or "person_boxes" not in person_results:
                    print("Invalid person results")
                    continue

                person_boxes = person_results["person_boxes"]
                track_ids = person_results["track_ids"]
                print("Person Boxes Detected:", person_boxes)
                print("Track IDs Detected:", track_ids)

                people_data = []
                for person_box, track_id in zip(person_boxes, track_ids):
                    x1, y1, x2, y2 = [int(coord) for coord in person_box]
                    frame_height, frame_width, _ = frame.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

                    person_image = frame[y1:y2, x1:x2]
                    if person_image.size == 0:
                        print(f"Empty image for track_id: {track_id}")
                        continue

                    person = {'bbox': [x1, y1, x2, y2], 'track_id': track_id, 'face_flag': "UNKNOWN", 'id_card': 'none'}
                    person_image = preprocess_frame(person_image)

                    try:
                        person_flag, face_score, face_boxes = process_faces(person_image)
                        print(f"Face Recognition Output (ID {track_id}):", person_flag, face_score, face_boxes)

                        if face_boxes:
                            person['face_flag'] = update_recognition_memory(track_id, person_flag, face_score)
                    except Exception as e:
                        print(f"Face recognition error: {e}")

                    try:
                        id_flag, id_box, id_card = detect_id_card(person_image)
                        print(f"ID Card Detection Output (ID {track_id}):", id_flag, id_box, id_card)
                        person['id_card'] = id_card
                    except Exception as e:
                        print(f"ID card detection error: {e}")

                    if person['face_flag'] != "UNKNOWN":
                        people_data.append(person)

                print("Final People Data for Annotation:", people_data)
                draw_annotations(frame, people_data)
            except Exception as e:
                print(f"Error during frame processing: {e}")

            img = cv2.resize(frame, (1280, 960))
            cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
