import threading
import time
import os
import json
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

from .face_alignment.alignment import norm_crop
from .face_detection.scrfd.detector import SCRFD
from .face_recognition.arcface.model import iresnet_inference
from .face_recognition.arcface.utils import compare_encodings, read_features
from .face_tracking.tracker.byte_tracker import BYTETracker
from .face_tracking.tracker.visualize import plot_tracking

# from face_detection.yolov11.detector import YOLODetector


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your models and move them to GPU
detector = SCRFD(model_file="Face_recognition/face_detection/scrfd/weights/scrfd_10g_bnkps.onnx")
# detector = YOLODetector(model_path="face_detection/yolov11/models/best.pt")

recognizer = iresnet_inference(
    model_name="r100", path="Face_recognition/face_recognition/arcface/weights/arcface_r100.pth", device=device
)

images_names, images_embs = read_features(feature_path="Face_recognition/datasets/face_features/feature")

id_face_mapping = {}

data_mapping = {
    "raw_image": [],
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": [],
}

stop_event = threading.Event()

def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def process_tracking(frame, tracker, args, frame_id, fps):
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

    tracking_tlwhs = []
    tracking_ids = []
    tracking_scores = []
    tracking_bboxes = []

    if outputs is not None:
        online_targets = tracker.update(
            outputs, [img_info["height"], img_info["width"]], (128, 128)
        )

        for i in range(len(online_targets)):
            t = online_targets[i]
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                x1, y1, w, h = tlwh
                tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(tid)
                tracking_scores.append(t.score)

        tracking_image = img_info["raw_img"]
    else:
        tracking_image = img_info["raw_img"]

    data_mapping["raw_image"] = img_info["raw_img"]
    data_mapping["detection_bboxes"] = bboxes
    data_mapping["detection_landmarks"] = landmarks
    data_mapping["tracking_ids"] = tracking_ids
    data_mapping["tracking_bboxes"] = tracking_bboxes

    return tracking_image

@torch.no_grad()
def get_feature(face_image):
    """Extract features from a face image."""
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)
    emb_img_face = recognizer(face_image).cpu().numpy()
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)

    return images_emb

def recognition(face_image):
    """Recognize a face image."""
    query_emb = get_feature(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    score = score[0]

    return score, name

def mapping_bbox(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
        0, y_max_inter - y_min_inter + 1
    )

    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area
    return iou

# Global variable to track unknown faces
unknown_faces_buffer = {}
unknown_faces_counter = {}

def recognize_face(frame):
    global unknown_faces_buffer, unknown_faces_counter

    # Load configuration
    file_name = "Face_recognition/face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)

    # Initialize tracker
    tracker = BYTETracker(args=config_tracking, frame_rate=30)
    frame_id = 0
    fps = 30

    modified_frame = process_tracking(frame, tracker, config_tracking, frame_id, fps)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Process each tracking box
    for i in range(len(data_mapping["tracking_bboxes"])):
        for j in range(len(data_mapping["detection_bboxes"])):
            mapping_score = mapping_bbox(box1=data_mapping["tracking_bboxes"][i],
                                         box2=data_mapping["detection_bboxes"][j])
            if mapping_score > 0.9:
                # Align and normalize face image for recognition
                face_alignment = norm_crop(img=frame, landmark=data_mapping["detection_landmarks"][j])
                score, name = recognition(face_image=face_alignment)

                if name is not None and score >= 0.5:
                    caption = f"SIETian"
                    color = (255, 0, 255)
                else:
                    caption = "UNKNOWN"
                    color = (0, 165, 255)

                    # Handle unknown face detection with a timer
                    if i not in unknown_faces_counter:
                        unknown_faces_counter[i] = time.time()
                    elif time.time() - unknown_faces_counter[i] >= 5:  # Confirm after 3 seconds
                        save_unknown_face(frame, current_time)
                        del unknown_faces_counter[i]

                x1, y1, x2, y2 = data_mapping["tracking_bboxes"][i]
                cv2.rectangle(modified_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(modified_frame, caption, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2)

                data_mapping["detection_bboxes"] = np.delete(data_mapping["detection_bboxes"], j, axis=0)
                data_mapping["detection_landmarks"] = np.delete(data_mapping["detection_landmarks"], j, axis=0)
                break

    return modified_frame

def save_unknown_face(frame, timestamp):
    camera_id = "Camera_1"  # Replace with actual camera ID if available
    filename_base = f"unknown_faces/frame_{timestamp.replace(':', '-')}.jpeg"

    # Save frame
    cv2.imwrite(filename_base, frame)

    # Create JSON metadata
    metadata = {
        "timestamp": timestamp,
        "camera_id": camera_id,
        "frame": filename_base,
    }

    json_filename = filename_base.replace('.jpeg', '.json')

    with open(json_filename, 'w') as json_file:
        json.dump(metadata, json_file)

