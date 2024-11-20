import cv2
from detector import YOLOv8Detector


# Load an example image
image_path = "test.jpg"
image = cv2.imread(image_path)

# Initialize detector with correct path
detector = YOLOv8Detector(model_file="weights/yolov8n-face.onnx")

# Run detection
outputs = detector.detect(image)
print(f"Outputs: {outputs}")
for pred in outputs:
    print(f"Pred: {pred}, Shape: {pred.shape}")
