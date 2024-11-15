import cv2

username = "aiml"
password = "Siet@2727"
camera_ip = "192.168.3.148"
port = "554"  # Default RTSP port for Hikvision cameras

# Construct the RTSP URL
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:{port}/Streaming/Channels/101"
cap = cv2.VideoCapture(f"ffmpeg -rtsp_transport tcp -i {rtsp_url}", cv2.CAP_FFMPEG)


if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
else:
    print("RTSP stream opened successfully.")

# Continuously capture frames from the stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    # Display the frame
    cv2.imshow('RTSP Stream', frame)

    # Press 'q' to exit the stream display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()