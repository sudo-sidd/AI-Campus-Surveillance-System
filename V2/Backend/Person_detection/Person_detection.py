import cv2
from ultralytics import YOLO
import os
import mediapipe as mp
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(BASE_DIR,"model", "person_detection.pt")

# Initialize YOLO and MediaPipe
yolo = YOLO(yolo_path)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def is_person_facing_camera(landmarks):
    """
    Filter out people at extreme angles and distances.
    Returns True if person is at an acceptable angle and distance.
    """
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    
    shoulder_depth_diff = abs(left_shoulder.z - right_shoulder.z)
    
    MAX_ANGLE_DEPTH_DIFF = 0.15    
    MIN_SHOULDER_WIDTH = 0.1       
    MAX_SHOULDER_WIDTH = 0.8       
    
    is_valid = (
        shoulder_depth_diff < MAX_ANGLE_DEPTH_DIFF and    # Not at extreme angle
        MIN_SHOULDER_WIDTH < shoulder_width and           # Not too far
        shoulder_width < MAX_SHOULDER_WIDTH               # Not too close
    )
    
    return is_valid

def track_persons(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    results = yolo.track(frame, persist=True, classes=[0], agnostic_nms=True, conf=0.5)  # Added track() method

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
    for bbox, track_id in zip(person_bboxes, track_ids):
        center_x, center_y, width, height = bbox
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        person_roi = frame[max(0, y1):min(frame_height, y2), 
                         max(0, x1):min(frame_width, x2)]
        if person_roi.size == 0:
            continue
        roi_results = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
        
        if roi_results.pose_landmarks and is_person_facing_camera(roi_results.pose_landmarks):
            filtered_bboxes.append([x1, y1, x2, y2])
            filtered_track_ids.append(track_id)
        
    return {
        "modified_frame": frame,
        "person_boxes": filtered_bboxes,
        "track_ids": filtered_track_ids,
    }


