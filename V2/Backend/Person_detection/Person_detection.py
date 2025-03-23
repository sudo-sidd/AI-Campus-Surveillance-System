# Distance filtering with  BBoxes
import cv2
from ultralytics import YOLO
import os
import mediapipe as mp
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(BASE_DIR,"model", "person_detection.pt")

yolo = YOLO(yolo_path)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def track_persons(frame):
    results = yolo.track(frame, persist=True, classes=[0], agnostic_nms=True, conf=0.5)
    
    if not results[0].boxes:
        return {
            "modified_frame": frame,
            "person_boxes": [],
            "track_ids": []
        }
    
    person_bboxes = results[0].boxes.xywh.cpu().numpy()
    
    if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
    else:
        track_ids = list(range(len(person_bboxes)))
    
    frame_height, frame_width = frame.shape[:2]
    filtered_bboxes = []
    filtered_track_ids = []
    
    MIN_HEIGHT_RATIO = 0.4
    MAX_HEIGHT_RATIO = 0.8 
    
    for bbox, track_id in zip(person_bboxes, track_ids):
        center_x, center_y, width, height = bbox
        
        height_ratio = height / frame_height
        
        if MIN_HEIGHT_RATIO < height_ratio < MAX_HEIGHT_RATIO:
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)
            
            filtered_bboxes.append([x1, y1, x2, y2])
            filtered_track_ids.append(track_id)
    
    return {
        "modified_frame": frame,
        "person_boxes": filtered_bboxes,
        "track_ids": filtered_track_ids,
    }
