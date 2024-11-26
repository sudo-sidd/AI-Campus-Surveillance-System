import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms

from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO model initialization
yolo_model = YOLO("face_detection/yolo/weights/yolo11n-face.pt")

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


def recognize_face(face_image):
    """Match face image against known embeddings."""
    query_emb = get_face_embedding(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    return score, images_names[id_min] if score > 0.5 else None


def process_frame(frame):
    """Detect faces, recognize them, and annotate the frame."""
    results = yolo_model.predict(frame, conf=0.5)
    for result in results:
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            face_image = frame[y1:y2, x1:x2]

            # Recognize face
            score, name = recognize_face(face_image)

            # Annotate frame
            if isinstance(score, np.ndarray):
                score_value = np.max(score)  # Or another metric like np.mean(score)
            else:
                score_value = score
            label = f"{name} ({score_value:.2f})" if name else "Unknown"
            color = (0, 255, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
    return frame


# Main function (example usage)
if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)  # Replace with video file if needed
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     annotated_frame = process_frame(frame)
    #     cv2.imshow("Face Recognition", annotated_frame)
    #
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
    frame = cv2.imread("test.jpg")
    modified_frame = process_frame(frame)
    cv2.imwrite("out.jpg", modified_frame)