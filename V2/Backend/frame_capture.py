import cv2
import os

# RTSP stream URL
rtsp_url = "rtsp://aiml:Siet@2727@192.168.3.183:554/Streaming/Channels/101"

# Directory to save frames
save_dir = "Face_recognition/raw_frames"
os.makedirs(save_dir, exist_ok=True)

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to open RTSP stream")
    exit()

frame_count = 0
save_every_n_frames = 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    frame_count += 1

    if frame_count % save_every_n_frames == 0:
        frame_filename = os.path.join(save_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")

    # Display the frame (optional)
    cv2.imshow('RTSP Stream', frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()