import cv2
import torch
from PIL import Image
import numpy as np
from scrfd import SCRFD  # Assuming SCRFD is a class for detection

def preprocess_frame(frame):
    """
    Preprocess a frame (image) for the model.
    Convert it to RGB and resize it if necessary.
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return Image.fromarray(image)  # Convert to PIL Image

def draw_bboxes(frame, faces):
    """
    Draw bounding boxes and keypoints on the frame.
    """
    for face in faces:
        # Extract bounding box and score
        bbox = face.bbox
        keypoints = face.keypoints
        prob = face.probability

        # Draw bounding box
        cv2.rectangle(
            frame,
            (int(bbox.upper_left.x), int(bbox.upper_left.y)),
            (int(bbox.lower_right.x), int(bbox.lower_right.y)),
            (0, 255, 0),  # Green box
            2,
        )

        # Draw keypoints
        for point in [keypoints.left_eye, keypoints.right_eye, keypoints.nose, keypoints.left_mouth, keypoints.right_mouth]:
            cv2.circle(frame, (int(point.x), int(point.y)), 3, (0, 0, 255), -1)  # Red keypoints

        # Put probability text
        cv2.putText(
            frame,
            f"{prob:.2f}",
            (int(bbox.upper_left.x), int(bbox.upper_left.y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1,
        )
    return frame

# Initialize SCRFD object
scrfd = SCRFD()  # Initialize without the model path argument

# Load the model
model_path = "face_detection/scrfd/weights/model.pth"  # Update with your .pth file path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scrfd.load_model(model_path, device)

# Open the webcam
cap = cv2.VideoCapture(0)  # Change index if you have multiple cameras

print("Starting webcam. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    image = preprocess_frame(frame)

    # Detect faces
    faces = scrfd.detect(image)

    # Draw bounding boxes and keypoints on the frame
    frame = draw_bboxes(frame, faces)

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
