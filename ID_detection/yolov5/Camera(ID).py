import torch
import cv2
import numpy as np

# Load the YOLOv5 model from the local repository with the custom weights
model = torch.hub.load('../yolov5', 'custom', path='best.pt', source='local')

# Set the model to evaluation mode
model.eval()

# Set up the webcam feed (usually 0 is the default camera, change if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

# Process the video stream frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to RGB as YOLOv5 expects RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference on the frame
    results = model(frame_rgb)

    # Render the results on the frame (this draws bounding boxes and labels)
    results.render()

    # Convert back to BGR to display with OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Show the processed frame
    cv2.imshow('ID Card Detection', frame_bgr)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()