import cv2
import numpy as np
from Person_detection.Person_detection import track_persons
from Face_recognition.face_recognize_yolo import process_faces
from ID_detection.yolov11.ID_Detection_test import detect_id_card

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


cap = cv2.VideoCapture(0)  # Change to RTSP stream URL if needed

if not cap.isOpened():
    print(f"Failed to open camera")
    exit()

frame_count = 0
process_every_n_frames = 5  # Process every N frames

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
                    if frame is None or person_image is None:
                        print(f"Empty or invalid image for track_id: {track_id}")
                        continue

                    # Initialize person data
                    person = {
                        'bbox': [x1, y1, x2, y2],
                        'track_id': track_id,
                        'face_flag': "UNKNOWN",
                        'face_box': [0, 0, 0, 0],
                        'id_flag': False,
                        'id_card':'none',
                        'id_box': [0, 0, 0, 0]
                    }

                    # Face recognition
                    try:
                        person_flag, face_box = process_faces(frame)
                        person['face_flag'] = person_flag
                        person['face_box'] = face_box
                        print(f"Face recognition successful for track_id {track_id}")
                    except Exception as e:
                        print(f"Error during face recognition for track_id {track_id}: {e}")

                    # ID card detection (uncomment if you want to enable this)
                    try:
                        id_flag , id_box , id_card = detect_id_card(person_image)
                        person['id_flag'] = id_flag
                        person['id_box'] = id_box
                        person['id_card'] = id_card
                        print(f"ID card detection successful for track_id {track_id}")
                    except Exception as e:
                        print(f"Error during ID card detection for track_id {track_id}: {e}")

                    # Append person data
                    people_data.append(person)

                # Draw bounding boxes and annotations
                draw_annotations(frame, people_data)

                print("people data:", people_data)

            except Exception as e:
                print(f"Error during frame processing: {e}")

            # Show the frame
            cv2.imshow('frame', frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
