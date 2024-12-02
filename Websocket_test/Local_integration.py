import cv2
from Face_recognition.face_recognize_yolo_updated import recognize_faces_in_persons
from Face_recognition.FaceTracker import FaceTracker
from ID_detection.yolov11.ID_Detection import detect_id_card
import time


camera_data = [{
        "camera_ip": "rtsp://aiml:Siet@2727@192.168.3.183:554/Streaming/Channels/101",
        "camera_location": "corridor"
    }]
current_frames = {}
camera_trackers = {}# Function to capture frames from each camera
def capture_frame(camera_index, camera_ip):
    global camera_trackers
    if camera_index not in camera_trackers:
        camera_trackers[camera_index] = FaceTracker()

    face_tracker = camera_trackers[camera_index]
    cap = cv2.VideoCapture(camera_ip)  # RTSP stream URL

    if not cap.isOpened():
        print(f"Failed to open camera {camera_index} at {camera_ip}")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            if frame_count % 1 == 0:
                # Process the frame for face and ID detection
                modified_frame, person_boxes, associations = detect_id_card(frame)
                modified_frame, flags  = recognize_faces_in_persons(modified_frame, person_boxes,face_tracker)

                cv2.imshow("Result", modified_frame)

            # _, jpeg = cv2.imencode('.jpg', frame)
            # current_frames[camera_index] = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        else:
            print(f"Error capturing frame from camera {camera_index}")

