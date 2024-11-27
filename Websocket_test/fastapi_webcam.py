import asyncio
import base64
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Create FastAPI instance
app = FastAPI()

# List to hold active WebSocket clients
websocket_clients = []

# Define camera IDs
camera_ids = [0, 1]  # Add more camera indices as needed

# Dictionary to store video capture objects for each camera
cap_dict = {}

# Async function to capture video from a specific camera and send frames to connected WebSocket clients
async def capture_video(camera_id: int):
    # Open the webcam (camera_id is used to identify the camera)
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Camera {camera_id} could not be opened.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode the frame to JPEG and then to base64 string
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')

        # Send the encoded frame to all connected WebSocket clients
        # We use asyncio.create_task to schedule the send task in the same event loop
        await asyncio.gather(*(send_frame_to_clients(websocket, camera_id, encoded_frame) for websocket in websocket_clients))

        # Sleep to limit the frame rate (for example, to 15 FPS)
        await asyncio.sleep(1 / 15)

    cap.release()

# Function to send frame data to a WebSocket client
async def send_frame_to_clients(websocket: WebSocket, camera_id: int, frame: str):
    await websocket.send_json({
        "camera_id": camera_id,  # Include camera_id in the message to distinguish different cameras
        "frame": frame
    })

# FastAPI startup event to start the video capture for all cameras
@app.on_event("startup")
async def on_startup():
    # Start the video capture for each camera using async functions
    for camera_id in camera_ids:
        # Store the video capture object in the dictionary
        cap_dict[camera_id] = cv2.VideoCapture(camera_id)
        asyncio.create_task(capture_video(camera_id))

# WebSocket endpoint to handle client connections and send video feed
@app.websocket("/ws/video/")
async def websocket_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection
    await websocket.accept()

    # Add the WebSocket client to the list
    websocket_clients.append(websocket)
    
    try:
        while True:
            # Wait for incoming message (you can use it for handling client requests)
            data = await websocket.receive_text()

    except WebSocketDisconnect:
        # Remove the client when they disconnect
        websocket_clients.remove(websocket)

# HTTP endpoint to serve the HTML page for video stream
@app.get("/")
async def get():
    return HTMLResponse("""
        <html>
            <head>
                <title>Live Webcam Feed</title>
            </head>
            <body>
                <h1>Live Webcam Feed</h1>
                <div id="video-feed"></div>
                <script>
                    const videoSocket = new WebSocket("ws://192.168.143.86:8000/ws/video/");  // Use your local server IP here

                    let cameraFeeds = {};  // Dictionary to store video feeds

                    videoSocket.onmessage = function(e) {
                        const data = JSON.parse(e.data);
                        const cameraId = data['camera_id']; // Camera ID
                        const frame = data['frame']; // Base64 frame from the server

                        // Create an image element if it doesn't exist
                        if (!cameraFeeds[cameraId]) {
                            cameraFeeds[cameraId] = document.createElement('img');
                            cameraFeeds[cameraId].id = `camera-feed-${cameraId}`;
                            document.getElementById('video-feed').appendChild(cameraFeeds[cameraId]);
                        }

                        // Update the image's src attribute to display the new frame
                        cameraFeeds[cameraId].src = 'data:image/jpeg;base64,' + frame;
                    };

                    videoSocket.onopen = function() {
                        console.log("WebSocket connection established.");
                    };
                    
                    videoSocket.onclose = function() {
                        console.log("WebSocket connection closed.");
                    };
                </script>
            </body>
        </html>
    """)

# Run the FastAPI server (use Uvicorn to run the app)
