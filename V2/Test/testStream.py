from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import time
import threading
import io

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
current_frame = None
frame_lock = threading.Lock()
camera_running = True

# Background video capture function
def background_capture(camera_id=0):
    global current_frame, camera_running
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Could not open camera with ID {camera_id}")
        return
    
    try:
        while camera_running:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue
                
            # Update the current frame with thread safety
            with frame_lock:
                current_frame = frame.copy()
            
            # Small delay to control CPU usage
            time.sleep(0.03)  # ~30 FPS
    except Exception as e:
        print(f"Error in background_capture: {e}")
    finally:
        cap.release()
        print("Camera released")

# Start the background capture thread
capture_thread = threading.Thread(target=background_capture, args=(0,))
capture_thread.daemon = True
capture_thread.start()

@app.get("/video")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

def generate_frames():
    """Generator function that yields the current frame for streaming."""
    global current_frame
    
    try:
        while camera_running:
            # Get the current frame with thread safety
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = current_frame.copy()
            
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to control frame rate
            time.sleep(0.03)
    except Exception as e:
        print(f"Error in generate_frames: {e}")

# Serve HTML content directly
@app.get("/")
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Video Stream</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #333;
            }
            .stream-img {
                width: 100%;
                max-height: 500px;
                border: 2px solid #333;
                object-fit: contain;
            }
        </style>
    </head>
    <body>
        <h1>Live Webcam Feed</h1>
        <div>
            <img class="stream-img" src="/video" alt="Live Stream" />
        </div>
    </body>
    </html>
    """
    return StreamingResponse(iter([html_content]), media_type="text/html")

# Cleanup when application stops
@app.on_event("shutdown")
def shutdown_event():
    global camera_running
    camera_running = False
    if capture_thread.is_alive():
        capture_thread.join(timeout=1.0)
    print("Application shutdown")

# For testing directly with this script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)