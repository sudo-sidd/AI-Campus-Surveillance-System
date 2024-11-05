import cv2
from Face_recognition.face_recognize import recognize_face
from ID_detection.yolov5.ID_Detection import detect_id_card

def video_feed():
    cap = cv2.VideoCapture(0)  # Access the default camera
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image.")
            break

        # Apply face recognition and ID card detection processing
        modified_frame = detect_id_card(frame)
        modified_frame = recognize_face(modified_frame)

        # Display the frame
        cv2.imshow('Video Feed', modified_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_feed()
