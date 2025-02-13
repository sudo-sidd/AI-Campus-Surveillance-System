import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from ultralytics import YOLO
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize YOLO face detection model
yolo_model = YOLO(os.path.join(BASE_DIR, "face_detection", "yolo", "weights", "yolo11n-face.pt"))

# Initialize ArcFace recognition model
recognizer = iresnet_inference(
    model_name="r100",
    path=os.path.join(BASE_DIR, "face_recognition", "arcface", "weights", "glink360k_cosface_r100_fp16_0.1.pth"),
    device=device,
)

# Load pre-saved face features
feature_path = os.path.join(BASE_DIR, "datasets", "face_features", "new")
images_name_path = os.path.join(feature_path, "images_name.npy")
images_emb_path = os.path.join(feature_path, "images_emb.npy")
images_names = np.load(images_name_path)
images_embs = np.load(images_emb_path)

# Store recognition history to stabilize identity tracking
face_memory = {}


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
    return emb_img_face / np.linalg.norm(emb_img_face)


def recognize_face(face_image):
    """Recognize a face image."""
    query_emb = get_feature(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    return score[0], name


def align_face(face_image):
    """Center crop and resize face for better alignment."""
    h, w = face_image.shape[:2]
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    aligned = face_image[y_start:y_start + size, x_start:x_start + size]
    return cv2.resize(aligned, (112, 112))


def enhance_face_quality(face_image):
    """Improve face image quality using CLAHE and denoising."""
    lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # Edge-preserving denoising
    denoised = cv2.bilateralFilter(l_enhanced, 9, 75, 75)

    merged = cv2.merge([denoised, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def process_faces(frame):
    """Process all detected faces with YOLO and ArcFace."""
    face_results = yolo_model.predict(frame, conf=0.7)
    recognized_faces = []

    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

    for result in face_results:
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox[:4])

            # # Skip small detections
            # if (x2 - x1) < 100 or (y2 - y1) < 100:
            #     continue

            try:
                cropped_face = frame[y1:y2, x1:x2]
                aligned_face = align_face(cropped_face)
                enhanced_face = enhance_face_quality(aligned_face)

                # Get recognition score
                raw_score, name = recognize_face(enhanced_face)

                # Calculate position-based score
                face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                distance = np.sqrt((face_center[0] - frame_center[0]) ** 2 +
                                   (face_center[1] - frame_center[1]) ** 2)
                position_score = 1 / (1 + distance / 100)  # Normalized position score

                # Combine scores with weighted average
                composite_score = raw_score * 0.7 + position_score * 0.3

                # Adaptive threshold
                dynamic_threshold = 0.50 if distance < 150 else 0.60

                if composite_score >= dynamic_threshold:
                    recognized_faces.append((name, composite_score, (x1, y1, x2, y2)))
                else:
                    recognized_faces.append(("UNKNOWN", 0.0, (x1, y1, x2, y2)))

            except Exception as e:
                print(f"Face processing error: {e}")
                continue

    return recognized_faces


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process faces once per frame
        detected_faces = process_faces(frame)

        for name, score, bbox in detected_faces:
            x1, y1, x2, y2 = bbox

            if name not in face_memory or score > face_memory[name]["score"]:
                face_memory[name] = {"score": score, "bbox": bbox}

            if score >= 0.5:
                label = f"{name} ({score:.2f})"
                color = (255, 0, 255)  # Magenta for recognized
            else:
                label = "UNKNOWN"
                color = (0, 165, 255)  # Orange for unknown

            # Draw face box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
