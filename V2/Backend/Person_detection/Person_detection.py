# Distance filtering 
 
# import cv2
# from ultralytics import YOLO
# import os
# import mediapipe as mp
# import numpy as np

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# yolo_path = os.path.join(BASE_DIR,"model", "person_detection.pt")

# # Initialize YOLO and MediaPipe
# yolo = YOLO(yolo_path)
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def track_persons(frame):
#     results = yolo.track(frame, persist=True, classes=[0], agnostic_nms=True, conf=0.5)
    
#     if not results[0].boxes:
#         return {
#             "modified_frame": frame,
#             "person_boxes": [],
#             "track_ids": []
#         }
    
#     person_bboxes = results[0].boxes.xywh.cpu().numpy()
    
#     if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
#         track_ids = results[0].boxes.id.int().cpu().tolist()
#     else:
#         track_ids = list(range(len(person_bboxes)))
    
#     frame_height, frame_width = frame.shape[:2]
#     filtered_bboxes = []
#     filtered_track_ids = []
    
#     # Define distance thresholds based on bounding box height
#     # These values need to be calibrated for your specific setup
#     MIN_HEIGHT_RATIO = 0.3  # Person must take up at least 20% of frame height to be "not too far"
#     MAX_HEIGHT_RATIO = 0.8  # Person must take up at most 80% of frame height to be "not too close"
    
#     for bbox, track_id in zip(person_bboxes, track_ids):
#         center_x, center_y, width, height = bbox
        
#         # Calculate height ratio as proxy for distance
#         height_ratio = height / frame_height
        
#         # Check if person is in the desired distance range
#         if MIN_HEIGHT_RATIO < height_ratio < MAX_HEIGHT_RATIO:
#             x1 = int(center_x - width / 2)
#             y1 = int(center_y - height / 2)
#             x2 = int(center_x + width / 2)
#             y2 = int(center_y + height / 2)
            
#             filtered_bboxes.append([x1, y1, x2, y2])
#             filtered_track_ids.append(track_id)
    
#     return {
#         "modified_frame": frame,
#         "person_boxes": filtered_bboxes,
#         "track_ids": filtered_track_ids,
#     }

# #Distance filtering + pose estimation

import cv2
from ultralytics import YOLO
import torch
import os
import mediapipe as mp
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(BASE_DIR, "model", "person_detection.pt")

# Initialize YOLO and MediaPipe
yolo = YOLO(yolo_path)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0  # Using lightweight model for better performance
)
mp_drawing = mp.solutions.drawing_utils

def is_person_facing_camera(landmarks):
    """
    Determine if a person is facing the camera based on landmark positions.
    
    Args:
        landmarks: MediaPipe pose landmarks
        
    Returns:
        Boolean indicating if the person is facing the camera
    """
    if landmarks is None:
        return False
        
    # Get key landmarks
    try:
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Check shoulder width vs depth
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        shoulder_depth_diff = abs(left_shoulder.z - right_shoulder.z) if hasattr(left_shoulder, 'z') else 0
        
        # Check hip width vs depth
        hip_width = abs(left_hip.x - right_hip.x)
        hip_depth_diff = abs(left_hip.z - right_hip.z) if hasattr(left_hip, 'z') else 0
        
        # Thresholds for determining if facing camera
        # Shoulders and hips should be roughly horizontal (similar z values)
        MAX_DEPTH_DIFF = 0.15
        MIN_WIDTH_RATIO = 0.05  # Minimum width relative to image width
        
        is_facing = (
            shoulder_depth_diff < MAX_DEPTH_DIFF and
            hip_depth_diff < MAX_DEPTH_DIFF and
            shoulder_width > MIN_WIDTH_RATIO and
            hip_width > MIN_WIDTH_RATIO
        )
        
        return is_facing
    except:
        return False
    
def track_persons(frame, run_pose_estimation=False):
    """
    Track persons and optionally run pose estimation.
    
    Args:
        frame: Input video frame
        run_pose_estimation: Whether to run pose estimation (default: False)
        
    Returns:
        Dictionary with modified_frame, person_boxes, track_ids, and optionally pose_results
    """
    with torch.no_grad():
        results = yolo.track(frame, persist=True, classes=[0], agnostic_nms=True, conf=0.7, iou=0.45)    
    
    if not results[0].boxes:
        return {
            "modified_frame": frame,
            "person_boxes": [],
            "track_ids": [],
            "height_ratios": [],
            "pose_results": [] if run_pose_estimation else None
        }
    
    person_bboxes = results[0].boxes.xywh.cpu().numpy()
    
    if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
    else:
        track_ids = list(range(len(person_bboxes)))
    
    frame_height, frame_width = frame.shape[:2]
    filtered_bboxes = []
    filtered_track_ids = []
    height_ratios = []
    pose_results_list = [] if run_pose_estimation else None
    
    MIN_HEIGHT_RATIO = 0.4  # Person must take up at least 30% of frame height to be "not too far"
    MAX_HEIGHT_RATIO = 0.8  # Person must take up at most 80% of frame height to be "not too close"
    
    for bbox, track_id in zip(person_bboxes, track_ids):
        center_x, center_y, width, height = bbox
        
        height_ratio = height / frame_height
        
        if MIN_HEIGHT_RATIO < height_ratio < MAX_HEIGHT_RATIO:
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)
            
            if run_pose_estimation:
                person_roi = frame[max(0, y1):min(frame_height, y2), 
                                max(0, x1):min(frame_width, x2)]
                
                if person_roi.size > 0:
                    roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    pose_result = pose.process(roi_rgb) 
                    
                    if pose_result.pose_landmarks and is_person_facing_camera(pose_result.pose_landmarks):
                        landmarks = pose_result.pose_landmarks.landmark
                        
                        nose_conf = landmarks[mp_pose.PoseLandmark.NOSE].visibility
                        left_eye_conf = landmarks[mp_pose.PoseLandmark.LEFT_EYE].visibility
                        right_eye_conf = landmarks[mp_pose.PoseLandmark.RIGHT_EYE].visibility
                        left_ear_conf = landmarks[mp_pose.PoseLandmark.LEFT_EAR].visibility
                        right_ear_conf = landmarks[mp_pose.PoseLandmark.RIGHT_EAR].visibility
                        
                        left_eye_x = landmarks[mp_pose.PoseLandmark.LEFT_EYE].x
                        right_eye_x = landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x
                        left_ear_z = landmarks[mp_pose.PoseLandmark.LEFT_EAR].z if hasattr(landmarks[mp_pose.PoseLandmark.LEFT_EAR], 'z') else 0
                        right_ear_z = landmarks[mp_pose.PoseLandmark.RIGHT_EAR].z if hasattr(landmarks[mp_pose.PoseLandmark.RIGHT_EAR], 'z') else 0
                        
                        eye_distance = abs(left_eye_x - right_eye_x)
                        ear_depth_diff = abs(left_ear_z - right_ear_z)
                        
                        if (nose_conf > VISIBILITY_THRESHOLD and
                            left_eye_conf > VISIBILITY_THRESHOLD and
                            right_eye_conf > VISIBILITY_THRESHOLD and
                            left_ear_conf > VISIBILITY_THRESHOLD and
                            right_ear_conf > VISIBILITY_THRESHOLD and
                            eye_distance > MIN_EYE_DISTANCE and
                            ear_depth_diff < MAX_EAR_DEPTH_DIFF):
                            
                            filtered_bboxes.append([x1, y1, x2, y2])
                            filtered_track_ids.append(track_id)
                            height_ratios.append(height_ratio)
                            pose_results_list.append({
                                "landmarks": pose_result.pose_landmarks,
                                "bbox": [x1, y1, x2, y2],
                                "facing_camera": True
                            })
            else:
                filtered_bboxes.append([x1, y1, x2, y2])
                filtered_track_ids.append(track_id)
                height_ratios.append(height_ratio)
    
    return {
        "modified_frame": frame,
        "person_boxes": filtered_bboxes,
        "track_ids": filtered_track_ids,
        "height_ratios": height_ratios,
        "pose_results": pose_results_list
    }


def draw_pose_landmarks(frame, pose_results):
    """
    Draw pose landmarks on the frame
    
    Args:
        frame: Input video frame
        pose_results: List of pose results from track_persons
        
    Returns:
        Frame with pose landmarks drawn
    """
    if pose_results is None:
        return frame
    
    frame_copy = frame.copy()
    
    for pose_data in pose_results:
        if pose_data is None or pose_data["landmarks"] is None:
            continue
            
        landmarks = pose_data["landmarks"]
        x1, y1, x2, y2 = pose_data["bbox"]
        
        # Extract person ROI
        person_roi = frame_copy[max(0, y1):min(frame.shape[0], y2), 
                              max(0, x1):min(frame.shape[1], x2)]
        
        if person_roi.size > 0:
            # Draw the landmarks on the ROI
            mp_drawing.draw_landmarks(
                person_roi, 
                landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            # Copy the landmark image back to the main frame
            frame_copy[max(0, y1):min(frame.shape[0], y2), 
                      max(0, x1):min(frame.shape[1], x2)] = person_roi
    
    return frame_copy




#BASE CODE
# import cv2
# from ultralytics import YOLO
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# yolo_path = os.path.join(BASE_DIR,"model", "person_detection.pt")

# yolo = YOLO(yolo_path)

# def track_persons(frame):
#     # Run YOLO tracking
#     results = yolo.track(frame, persist=True, iou=0.5, conf=0.7)

#     if not results[0].boxes:
#         return {"modified_frame": frame, "person_boxes": [], "track_ids": []}

#     # Get frame dimensions
#     frame_height, frame_width = frame.shape[:2]

#     # Bounding boxes (normalized or scaled by YOLO during processing)
#     person_bboxes = results[0].boxes.xywh.cpu().numpy()  # (center_x, center_y, width, height)
#     track_ids = results[0].boxes.id.int().cpu().tolist()  # Tracking IDs

#     # Convert bounding boxes to original frame size (x1, y1, x2, y2)
#     person_bboxes_scaled = []
#     for bbox in person_bboxes:
#         center_x, center_y, width, height = bbox
#         x1 = int((center_x - width / 2))
#         y1 = int((center_y - height / 2))
#         x2 = int((center_x + width / 2))
#         y2 = int((center_y + height / 2))
#         person_bboxes_scaled.append([x1, y1, x2, y2])

#     return {
#         "modified_frame": frame,
#         "person_boxes": person_bboxes_scaled,  # Scaled to match the original frame
#         "track_ids": track_ids,
#     }



