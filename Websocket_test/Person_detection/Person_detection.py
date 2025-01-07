import numpy as np
import cv2
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

yolo_path = os.path.join(BASE_DIR, "model", "best.pt")

# Initialize YOLO with the tracking model
yolo = YOLO(yolo_path)

def draw_track(frame, track):
    """Helper function to draw a single track"""
    bbox = track['bbox']
    track_id = track['id']
    x1, y1, x2, y2 = map(int, bbox)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Draw track ID
    text = f'ID: {track_id}'
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_width, y1), (255, 255, 255), -1)
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def track_persons(frame):
    """
    Track persons in the frame using YOLOv8's built-in tracking.
    Returns the modified frame with drawn tracks.
    """
    results = yolo.track(frame, persist=True)
    tracked_persons = []

    for result in results:
        for track in result.tracks:
            bbox = track['bbox']
            track_id = track['id']
            cls = track['cls']

            # Only consider 'person' class, which is typically class '0' in YOLO
            if int(cls) == 0:
                draw_track(frame, {'bbox': bbox, 'id': track_id})
                tracked_persons.append({
                    'id': track_id,
                    'bbox': bbox
                })

    return {
        'tracked_persons': tracked_persons,
        'modified_frame': frame
    }

if __name__ == "__main__":
    cap = cv2.VideoCapture('video.mp4')  # Replace 'video.mp4' with your video file path or 0 for webcam

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = track_persons(frame)

        cv2.imshow("Tracked Frame", result['modified_frame'])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
