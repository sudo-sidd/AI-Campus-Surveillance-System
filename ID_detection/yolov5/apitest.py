import requests
import cv2
import numpy as np
import io

def send_image_for_detection(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    try:
        response = requests.post('http://localhost:5001/detect', files={'frame': img_bytes})

        if response.status_code == 200:
            img_arr = np.frombuffer(response.content, np.uint8)
            return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        else:
            print(f"Error: {response.json()}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Face recognition API error: {e}")
        return None

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    # Send the current frame to the detection API
    detected_img = send_image_for_detection(frame)

    # If the API returns a valid image, display it, otherwise display the original frame
    if detected_img is not None:
        cv2.imshow('Detected Video', detected_img)
    else:
        cv2.imshow('Detected Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
