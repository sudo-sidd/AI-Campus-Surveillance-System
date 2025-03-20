# Face Recognition & ID Detection System

## Project Overview

This repository contains an advanced computer vision system that performs real-time face recognition and ID card detection. The system tracks individuals through video streams, identifies faces, detects ID cards, and provides a web-based interface for monitoring.

## Key Features

- **Person Detection & Tracking**: Uses YOLO to detect and track people with unique IDs
- **Pose Estimation Filtering**: Uses MediaPipe to filter individuals based on their orientation (facing camera)
- **Face Recognition**: Identifies known individuals from a database of faces
- **ID Card Detection**: Detects and reads ID cards shown to the camera
- **Real-time Processing**: Optimized for real-time video analysis
- **WebSocket Streaming**: Provides annotated video feed via WebSocket
- **Distance-based Filtering**: Filters out individuals too far or too close to the camera

## Technical Architecture

### Components

1. **Person Detection**: YOLO-based detector for locating people in frames
2. **Pose Estimation**: MediaPipe Pose for determining if a person is facing the camera
3. **Face Recognition**: Detects and matches faces against known database
4. **ID Card Detection**: Recognizes ID cards in the frame
5. **FastAPI Backend**: Serves detection results and processed video
6. **WebSocket Streaming**: Provides real-time video with annotations

### Project Structure

```
Face_rec-ID_detection/
├── V1/                   # Original implementation
├── V2/                   # Current implementation
│   ├── Backend/
│   │   ├── Person_detection/
│   │   │   ├── model/
│   │   │   ├── Person_detection.py
│   │   │   └── test_detection.py
│   │   ├── Face_recognition/
│   │   └── ID_detection/
│   ├── fastapi_stream.py # API endpoint for streaming
│   └── Testing_stream.py # Direct testing interface
└── .env                  # Environment configuration
```

## Installation & Setup

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Face_rec-ID_detection.git
cd Face_rec-ID_detection

# Set up virtual environment
python -m venv face-rec
source face-rec/bin/activate  # On Windows: face-rec\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model files (if not included in repository)
# Instructions specific to your model sources
```

### Configuration

1. Edit `.env` file to set your camera IP addresses
2. Configure detection thresholds in respective Python files
3. Add face recognition database images to appropriate directory

## Usage

### Testing Stream

Run the direct testing interface to process video locally:

```bash
python V2/Backend/Testing_stream.py
```

### FastAPI Server

Start the API server for web-based monitoring:

```bash
python V2/fastapi_stream.py
```

Then access the interface at `http://localhost:7000`

### WebSocket Connection

Connect to the WebSocket endpoint at:

```
ws://localhost:7000/ws/video/{camera_id}/
```

## Customization Options

### Person Detection Filtering

In `Person_detection.py`, adjust these parameters:

- `MIN_HEIGHT_RATIO` and `MAX_HEIGHT_RATIO`: Control detection distance
- `MAX_ANGLE_DEPTH_DIFF`: Control acceptable viewing angles
- `min_detection_confidence` in Pose model: Detection sensitivity

### Recognition Memory

In `Testing_stream.py`, modify:

- `frame_memory`: How many frames to remember a recognition
- `ema_alpha`: Weight for exponential moving average confidence scoring

### Processing Performance

- Adjust `process_every_n_frames` to process fewer frames for better performance
