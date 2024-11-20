import numpy as np
import cv2
from scipy.special import expit  # For sigmoid function, if needed
import onnxruntime as ort

class YOLOv8Detector:
    def __init__(self, model_file):
        self.session = ort.InferenceSession(model_file)

    def sigmoid(self, x):
        return expit(x)

    def post_process(self, outputs, image_shape):
        results = []
        for pred in outputs:
            # Assuming pred[4] is an array, use a specific index or condition
            confidence = pred[4]  # Extract confidence
            if isinstance(confidence, float) or isinstance(confidence, int):
                if confidence > 0.5:  # Adjust confidence threshold as needed
                    # Process further
                    results.append(pred)
            elif isinstance(confidence, np.ndarray):  # Handle arrays
                # Use any() or all(), or iterate over elements
                if (confidence > 0.5).any():  # Modify logic based on requirements
                    # Process further
                    results.append(pred)
            else:
                raise ValueError(f"Unexpected type for confidence: {type(confidence)}")
        return results

    def detect(self, image):
        input_shape = (1, 3, 640, 640)  # Adjust as needed
        image_resized = cv2.resize(image, (640, 640))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_transposed = np.transpose(image_normalized, (2, 0, 1)).reshape(input_shape)

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: image_transposed})

        bboxes, confidences, class_ids = self.post_process(outputs, image.shape)
        return {
            "bboxes": np.array(bboxes),
            "confidences": np.array(confidences),
            "class_ids": np.array(class_ids),
        }
