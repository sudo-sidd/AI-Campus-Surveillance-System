import cv2
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(BASE_DIR,"model", "person_detection.pt")

yolo = YOLO(yolo_path)

def track_persons(frame):
    # Run YOLO tracking
    results = yolo.track(frame, persist=True, iou=0.5, conf=0.5)

    if not results[0].boxes:
        return {"modified_frame": frame, "person_boxes": [], "track_ids": []}

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Bounding boxes (normalized or scaled by YOLO during processing)
    person_bboxes = results[0].boxes.xywh.cpu().numpy()  # (center_x, center_y, width, height)
    track_ids = results[0].boxes.id.int().cpu().tolist()  # Tracking IDs

    # Convert bounding boxes to original frame size (x1, y1, x2, y2)
    person_bboxes_scaled = []
    for bbox in person_bboxes:
        center_x, center_y, width, height = bbox
        x1 = int((center_x - width / 2))
        y1 = int((center_y - height / 2))
        x2 = int((center_x + width / 2))
        y2 = int((center_y + height / 2))
        person_bboxes_scaled.append([x1, y1, x2, y2])

    return {
        "modified_frame": frame,
        "person_boxes": person_bboxes_scaled,  # Scaled to match the original frame
        "track_ids": track_ids,
    }


