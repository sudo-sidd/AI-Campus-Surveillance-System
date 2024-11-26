import asyncio
import websockets
import json
import os
import cv2
import base64
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
from Face_recognition.face_recognize_yolo import recognize_faces_in_persons
from ID_detection.yolov11.ID_Detection import detect_id_card

# MongoDB connection
try:
    client = MongoClient(os.getenv("MONGO_URI", 'mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/'))
    db = client['ML_project']  # Replace with your database name
    collection = db['DatabaseDB']  # Replace with your collection name
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit(1)

def save_to_database(frame, person_box, flag, id_card_type, camera_id):
    """
    Save detection results to MongoDB for unidentified persons or students without ID cards.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    x1, y1, x2, y2 = [int(coord) for coord in person_box]
    person_image = frame[y1:y2, x1:x2]

    if flag in ["UNKNOWN", "SIETIAN"] and not id_card_type:
        image_name = f"person_{camera_id}_{current_time}.jpg"
        image_path = os.path.join("images", image_name)

        try:
            os.makedirs("images", exist_ok=True)  # Ensure directory exists
            cv2.imwrite(image_path, person_image)

            document = {
                "_id": ObjectId(),
                "Reg_no": None,  # Replace with actual ID if available
                "location": camera_id,
                "time": datetime.now(),
                "Role": "Unidentified" if flag == "UNKNOWN" else "Student",
                "Wearing_id_card": bool(id_card_type),
                "image": image_path,
                "recognition_status": "Unknown" if flag == "UNKNOWN" else "Recognized",
            }

            result = collection.insert_one(document)
            print(f"Document inserted with _id: {result.inserted_id}")
        except Exception as e:
            print(f"Error saving to database: {e}")

async def process_video_feed(websocket):
    """
    Process video feed, detect faces, and send results via WebSocket.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video feed.")
        await websocket.send(json.dumps({"error": "Could not access video feed"}))
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                await websocket.send(json.dumps({"error": "Failed to capture frame"}))
                break

            # Perform ID card detection and face recognition
            modified_frame, person_boxes, associations = detect_id_card(frame)
            modified_frame, flags = recognize_faces_in_persons(modified_frame, person_boxes)

            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', modified_frame)
            base64_frame = base64.b64encode(buffer).decode('utf-8')

            # Prepare and send results (detection data + frame)
            results = []
            for idx, (person_box, flag) in enumerate(zip(person_boxes, flags)):
                id_card_type = associations[idx] if idx < len(associations) else None
                results.append({
                    "box": person_box,
                    "status": flag,
                    "id_card_type": id_card_type,
                })

                # Save specific detections to the database
                save_to_database(frame, person_box, flag, id_card_type, camera_id=0)

            await websocket.send(json.dumps({
                "detections": results,
                "frame": base64_frame  # Send base64 encoded frame
            }))

            # Display the frame locally (optional)
            # cv2.imshow('Video Feed', modified_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            await asyncio.sleep(0.1)  # Smooth frame sending
    except websockets.ConnectionClosed:
        print("Client disconnected.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

async def websocket_server():
    """
    Start the WebSocket server to stream face recognition data.
    """
    async with websockets.serve(process_video_feed, "0.0.0.0", 8765):
        print("WebSocket server started at ws://0.0.0.0:8765")
        await asyncio.Future()  # Keep the server running

if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)  # Create directory for images
    asyncio.run(websocket_server())
