from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import uvicorn

app = FastAPI()

def get_camera_frames():
    """Generator function that yields video frames from webcam"""
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            # Read a frame from the webcam
            success, frame = cap.read()
            if not success:
                break
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            # Yield the frame as bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        # Release the webcam when the function exits
        cap.release()

@app.get("/frame")
def get_frame():
    """Return a single frame as a JPEG image"""
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        return {"error": "Failed to capture frame"}
    
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return {"error": "Failed to encode frame"}
    
    return StreamingResponse(
        iter([buffer.tobytes()]), 
        media_type="image/jpeg"
    )

@app.get("/video_feed")
def video_feed():
    """Return a streaming response with the video feed"""
    return StreamingResponse(
        get_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)