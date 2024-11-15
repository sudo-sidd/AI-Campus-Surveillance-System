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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
detector = SCRFD(model_file="Face_recognition/face_detection/scrfd/weights/scrfd_10g_bnkps.onnx")
recognizer = iresnet_inference(
    model_name="r100",
    path="Face_recognition/face_recognition/arcface/weights/arcface_r100.pth",
    device=device
)

images_names, images_embs = read_features(feature_path="Face_recognition/datasets/face_features/feature")


def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def is_face_in_person_box(face_box, person_box, iou_threshold=0.5):
    """Check if face box is within person box"""
    x1 = max(face_box[0], person_box[0])
    y1 = max(face_box[1], person_box[1])
    x2 = min(face_box[2], person_box[2])
    y2 = min(face_box[3], person_box[3])

    if x2 <= x1 or y2 <= y1:
        return False

    face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
    intersection = (x2 - x1) * (y2 - y1)

    return intersection / face_area > iou_threshold


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

    return {
        "raw_image": img_info["raw_img"],
        "detection_bboxes": bboxes,
        "detection_landmarks": landmarks,
        "tracking_ids": tracking_ids,
        "tracking_bboxes": tracking_bboxes
    }


@torch.no_grad()
def recognize_face(frame, person_boxes):
    """
    Process frame for face recognition within person boxes
    Returns:
    - Modified frame with annotations
    - List of flags corresponding to person boxes
    """
    # Load configuration
    file_name = "Face_recognition/face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)

    # Initialize tracker
    tracker = BYTETracker(args=config_tracking, frame_rate=30)
    frame_id = 0
    fps = 30

    # Process tracking
    tracking_data = process_tracking(frame, tracker, config_tracking, frame_id, fps)

    # Initialize flags for each person
    flags = ['UNKNOWN'] * len(person_boxes)  # Default to 'UNKNOWN'

    # Draw person boxes
    for i, person_box in enumerate(person_boxes):
        cv2.rectangle(frame,
                      (int(person_box[0]), int(person_box[1])),
                      (int(person_box[2]), int(person_box[3])),
                      (0, 255, 0), 2)  # Green for person boxes

    # If no faces detected, return original frame and empty flags
    if len(tracking_data["tracking_bboxes"]) == 0:
        return frame, flags

    # Process each tracked face
    for i, (face_box, face_id) in enumerate(zip(tracking_data["tracking_bboxes"],
                                                tracking_data["tracking_ids"])):
        # Find corresponding detection box and landmarks
        for j, (det_box, landmark) in enumerate(zip(tracking_data["detection_bboxes"],
                                                    tracking_data["detection_landmarks"])):
            if mapping_bbox(face_box, det_box) > 0.9:
                # Check which person box this face belongs to
                for person_idx, person_box in enumerate(person_boxes):
                    if is_face_in_person_box(face_box, person_box):
                        # Align and recognize face
                        face_alignment = norm_crop(img=frame, landmark=landmark)
                        score, name = recognition(face_image=face_alignment)

                        if name is not None and score >= 0.5:
                            flags[person_idx] = f"SIETIAN ({name})"  # Include recognized name
                            color = (255, 0, 255)  # Magenta for recognized
                        else:
                            flags[person_idx] = "UNKNOWN"
                            color = (0, 165, 255)  # Orange for unknown

                        # Draw face box and label
                        cv2.rectangle(frame,
                                      (int(face_box[0]), int(face_box[1])),
                                      (int(face_box[2]), int(face_box[3])),
                                      color, 2)
                        cv2.putText(frame, f"{flags[person_idx]} (ID: {face_id})",
                                    (int(face_box[0]), int(face_box[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        break

                # Remove processed detection
                tracking_data["detection_bboxes"] = np.delete(tracking_data["detection_bboxes"], j, axis=0)
                tracking_data["detection_landmarks"] = np.delete(tracking_data["detection_landmarks"], j, axis=0)
                break

    return frame, flags


def recognition(face_image):
    """Recognize a face image."""
    query_emb = get_feature(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    score = score[0]
    return score, name


@torch.no_grad()
def get_feature(face_image):
    """Extract features from a face image."""
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)
    emb_img_face = recognizer(face_image).cpu().numpy()
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb


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