import cv2
import os
import time
from datetime import datetime

RTSP_URL = "rtsp://aiml:Siet@2727@172.16.3.183:554/Streaming/Channels/101"

# Output directory to save frames
OUTPUT_DIR = "captured_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open the RTSP stream
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("Failed to open RTSP stream")
    exit()

frame_count = 0
current_second = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame")
        break

    # Get current time
    now = datetime.now()
    timestamp = now.strftime("%H-%M-%S")

    # Reset frame count if a new second starts
    if now.second != current_second:
        frame_count = 0
        current_second = now.second

    # Generate filename with time and frame count
    filename = f"{OUTPUT_DIR}/{timestamp}-{frame_count}.jpg"

    # Save the frame
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

    frame_count += 1

    # Display the frame (optional)
    cv2.imshow("RTSP Stream", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
