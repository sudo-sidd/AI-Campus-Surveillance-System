import cv2
import numpy as np
import torch
from ultralytics import YOLO
from Person_detection.Person_detection_test import track_persons

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize YOLO face detection model
face_model = YOLO("/mnt/data/PROJECTS/Face_rec-ID_detection/V2/Backend/Face_recognition/face_detection/yolo/weights/yolo11n-face.pt")


def draw_annotations(frame, person_data):
    """Draw bounding boxes and annotations for persons and faces."""
    for person in person_data:
        x1, y1, x2, y2 = person['bbox']
        track_id = person['track_id']

        # Draw person bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw face bounding box if detected
        if person['face_detected']:
            fx1, fy1, fx2, fy2 = person['face_bbox']
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

        # Add text annotations
        text = f"ID: {track_id} | Face: {'Yes' if person['face_detected'] else 'No'}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Add background for text
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (255, 255, 255), -1)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def detect_faces(person_image):
    """Detect faces in the person crop using YOLO."""
    results = face_model.predict(person_image, conf=0.5)

    if not results or len(results[0].boxes) == 0:
        return False, None

    # Get the first detected face (highest confidence)
    bbox = results[0].boxes.xyxy[0].cpu().numpy()
    return True, bbox.astype(int)


def process_frame(frame):
    """Process a single frame for person and face detection."""
    # Detect and track persons
    person_results = track_persons(frame)

    if not person_results or "person_boxes" not in person_results:
        return frame

    person_boxes = person_results["person_boxes"]
    track_ids = person_results["track_ids"]

    people_data = []

    # Process each detected person
    for person_box, track_id in zip(np.array(person_boxes).tolist(), track_ids):
        x1, y1, x2, y2 = [int(coord) for coord in person_box]

        # Validate and clip bounding boxes
        frame_height, frame_width, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

        # Crop person image
        person_image = frame[y1:y2, x1:x2]
        if person_image.size == 0:
            continue

        # Initialize person data
        person = {
            'bbox': [x1, y1, x2, y2],
            'track_id': track_id,
            'face_detected': False,
            'face_bbox': None
        }

        # Detect face in person crop
        try:
            face_detected, face_bbox = detect_faces(person_image)
            if face_detected:
                fx1, fy1, fx2, fy2 = face_bbox
                person['face_detected'] = True
                person['face_bbox'] = [
                    x1 + fx1,
                    y1 + fy1,
                    x1 + fx2,
                    y1 + fy2
                ]

                # cropped_face = frame[y1+fy1:y2+fy2, x1+fx1:x2+fx2]
                # cv2.imwrite(f'cropped_faces/{track_id}.jpg',cropped_face)
        except Exception as e:
            print(f"Face detection error: {e}")

        if person['face_detected'] == True:
                people_data.append(person)

    draw_annotations(frame, people_data)
    return frame

def main():
    cap = cv2.VideoCapture("rtsp://aiml:Siet@2727@192.168.3.183:554/Streaming/Channels/101")
    if not cap.isOpened():
        print("Failed to open camera")
        return

    frame_count = 0
    process_every_n_frames = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            try:
                processed_frame = process_frame(frame)
                # Resize for display
                display_frame = cv2.resize(processed_frame, (1280,720))
                cv2.imshow('Combined Detection', display_frame)
            except Exception as e:
                print(f"Error processing frame: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()