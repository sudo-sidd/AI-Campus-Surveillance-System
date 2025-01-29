import cv2
import numpy as np
from Person_detection.Person_detection import track_persons
from Face_recognition.face_recognize_yolo import process_faces
from ID_detection.yolov11.ID_Detection import detect_id_card

def draw_annotations(frame, person_data):
    """Draw bounding boxes and annotations on the frame."""
    for person in person_data:
        x1, y1, x2, y2 = person['bbox']
        track_id = person['track_id']
        face_flag = person['face_flag'][0][0]
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

                    # Face recognition
                    try:
                        person_flag, face_boxes = process_faces(person_image)
                        if face_boxes:
                            fb_x1, fb_y1, fb_x2, fb_y2 = face_boxes[0]
                            fb_x1 += x1
                            fb_y1 += y1
                            fb_x2 += x1
                            fb_y2 += y1
                            person['face_flag'] = person_flag
                            person['face_box'] = [fb_x1, fb_y1, fb_x2, fb_y2]
                        else:
                            person['face_flag'] = [("UNKNOWN", (0, 0))]
                    except Exception as e:
                        print(f"Face recognition error: {e}")

                    # ID card detection
                    try:
                        id_flag, id_box, id_card = detect_id_card(person_image)
                        person['id_flag'] = id_flag
                        person['id_box'] = id_box
                        person['id_card'] = id_card
                    except Exception as e:
                        print(f"ID detection error: {e}")

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
