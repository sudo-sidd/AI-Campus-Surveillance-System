import cv2
import numpy as np
import torch
from mtcnn import MTCNN
from torchvision import transforms
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
import os

# username = os.getenv("CAMERA_USERNAME", "aiml")
# password = os.getenv("CAMERA_PASSWORD", "Siet@2727")
# camera_ip = os.getenv("CAMERA_IP", "192.168.3.148")
# port = "554"  # Default RTSP port for Hikvision cameras
#
# # Construct the RTSP URL
# rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:{port}/Streaming/Channels/101"



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MTCNN detector
mtcnn_detector = MTCNN()

# ArcFace model initialization

recognizer = iresnet_inference(
    model_name="r100",
    path=os.path.join(
        BASE_DIR, "face_recognition", "arcface", "weights", "arcface_r100.pth"
    ),
    device=device,
)
# Preloaded face embeddings and names
images_names, images_embs = read_features(
    feature_path=os.path.join(
        BASE_DIR, "datasets", "face_features", "feature"
    )
)
# Preprocessing for ArcFace input
face_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])



def get_face_embedding(face_image):
    """Extract features for a given face image."""
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_tensor = face_preprocess(face_image).unsqueeze(0).to(device)
    emb = recognizer(face_tensor).cpu().detach().numpy()
    return emb / np.linalg.norm(emb)


def face_rec(face_image):
    """Match face image against known embeddings."""
    query_emb = get_face_embedding(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    return score, images_names[id_min] if score > 0.5 else None


def is_face_in_person_box(face_box, person_box, iou_threshold=0.5):
    """Check if a face bounding box is within a model bounding box."""
    x1 = max(face_box[0], person_box[0])
    y1 = max(face_box[1], person_box[1])
    x2 = min(face_box[2], person_box[2])
    y2 = min(face_box[3], person_box[3])

    if x2 <= x1 or y2 <= y1:
        return False

    face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
    intersection_area = (x2 - x1) * (y2 - y1)

    return intersection_area / face_area > iou_threshold
def recognize_faces_in_persons(frame):
    """
    Detect faces within model bounding boxes and classify them as SIETIAN, UNKNOWN, or UNDETERMINED.
    Args:
        frame: The video frame to process.
        person_bboxes: List of model bounding boxes in the format [(x1, y1), (x2, y2)].
    Returns:
        modified_frame: Frame with model bounding boxes and recognition results.
        states: List of states for each model box - SIETIAN, UNKNOWN, or UNDETERMINED.
    """
    # Detect faces using YOLO (or another detector)
    face_results = mtcnn_detector.detect_faces(frame)


    face_boxes = [face["box"] for face in face_results]



    for face_box in face_boxes:         # Crop and preprocess the face image
        x1, y1, x2, y2 = map(int, face_box)
        cropped_face = frame[y1:y2, x1:x2]
        score, name = face_rec(cropped_face)

                    # Update state and draw on the frame
        if name:
            state = f"SIETIAN ({name})"
            color = (0, 255, 0)  # Green for recognized
        else:
            state= "UNKNOWN"
            color = (0, 0, 255)  # Red for unknown

                    # Draw face bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                        frame,
                        f"{state}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )


        # If no face is detected in the model box


    return frame


# Main function (example usage)
if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     annotated_frame = recognize_faces_in_persons(frame)
    #     cv2.imshow("Face Recognition", annotated_frame)
    #
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    #
    # cap.release()
    img = cv2.imread("test.jpg")
    annotated_frame = recognize_faces_in_persons(img)
    cv2.imwrite("op.jpg",annotated_frame)


