# Distance filtering with  MediaPipe Pose
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
    
    # Define distance thresholds based on bounding box height
    # These values need to be calibrated for your specific setup
    MIN_HEIGHT_RATIO = 0.4  # Person must take up at least 20% of frame height to be "not too far"
    MAX_HEIGHT_RATIO = 0.8  # Person must take up at most 80% of frame height to be "not too close"
    
    for bbox, track_id in zip(person_bboxes, track_ids):
        center_x, center_y, width, height = bbox
        
        # Calculate height ratio as proxy for distance
        height_ratio = height / frame_height
        
        # Check if person is in the desired distance range
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



# # Distance filtering with  BBoxes
# import cv2
# from ultralytics import YOLO
# import os
# import mediapipe as mp
# import numpy as np

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# yolo_path = os.path.join(BASE_DIR,"model", "person_detection.pt")

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
    
#     MIN_HEIGHT_RATIO = 0.4
#     MAX_HEIGHT_RATIO = 0.8 
    
#     for bbox, track_id in zip(person_bboxes, track_ids):
#         center_x, center_y, width, height = bbox
        
#         height_ratio = height / frame_height
        
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


# #BASE CODE
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



