import cv2
from ultralytics import YOLO

# Load the YOLO model with the trained weights
model = YOLO("/mnt/data/Projects/CCTV_object-detection/ID+Person+face/ID_detection/yolov11/models/face/best.pt")  # Adjust this to the correct model path

# Set up the webcam feed (usually 0 is the default camera, change if needed)
cap = cv2.VideoCapture('/mnt/data/Projects/Label_ME/Screenrecorder-2024-11-12-10-39-58-745.mp4')

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

# Process the video stream frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference on the current frame
    results = model(frame)  # Perform inference with YOLO model

    # Loop over each detected object in the results
    for result in results[0].boxes:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = result.xyxy[0].tolist()  # Convert tensor to list of coordinates

        # Get the class index and map it to a class name
        class_id = int(result.cls)
        class_name = results[0].names[class_id]

        # Get the confidence score (probability)
        confidence = result.conf.item()  # Convert tensor to scalar value

        # Draw the bounding box and label with confidence
        # label = f"{class_name} ({confidence:.2f})"
        label = ""
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Draw label
    # Show the processed frame with bounding boxes and labels
    cv2.imshow('ID Card Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
