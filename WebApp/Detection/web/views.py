from django.http import JsonResponse
from django.shortcuts import render
from .mongo_connection import get_database
from bson import ObjectId
from django.http import HttpResponse, StreamingHttpResponse
import json
import os
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import requests

def my_view(request):
    db = get_database()
    collection = db['DatabaseDB']  # Replace with your actual collection name
    data = list(collection.find())  # Fetch data from the collection

    # Convert ObjectId to string
    for document in data:
        document['_id'] = str(document['_id'])

    return JsonResponse(data, safe=False)  # Return the modified data as a JSON response

def insert_document(request):
    db = get_database()
    from datetime import datetime, timezone
    collection = db['DatabaseDB']
    reg_no = 123456789012
    location = 'test'
    role = 'insider'
    wearing_id_card = input("Is wearing ID card? (true/false): ").strip().lower() == 'false'
    image = {}  # Replace with actual image data if available, or leave as empty dict

    # Generate random _id and get current time
    document = {
        "_id": ObjectId(),  # Generate a random ObjectId
        "Reg_no": reg_no,
        "location": location,
        "time": datetime.now(timezone.utc),  # Use current UTC time
        "Role": role,
        "Wearing_id_card": wearing_id_card,
        "image": image
    }

    collection.insert_one(document)
    return JsonResponse({"status": "success"})

def home(request):
    return render(request, 'home.html')

def detection_view(request):
    """
    View to categorize and display individuals based on their role and ID card status.
    """
    # Connect to the database
    db = get_database()
    collection = db['DatabaseDB']  # Replace with your actual collection name

    try:
        # Fetch all documents from the collection
        data = list(collection.find())

        # Initialize categories
        outsiders = []
        non_id_holders = []

        # Categorize data
        for document in data:
            document['_id'] = str(document['_id'])  # Convert ObjectId to string for JSON compatibility
            role = document.get('role', '').lower()  # Ensure role field is processed in lowercase

            # Categorize by role and ID card status
            if role == 'outsider':
                outsiders.append(document)
            elif not document.get('wearing_id_card', False):  # Check ID card status
                non_id_holders.append(document)

        # Debugging (Optional)
        print(f"Outsiders: {outsiders}\n")
        print(f"Non-ID Holders: {non_id_holders}\n")

    except Exception as e:
        # Handle any database-related errors gracefully
        print(f"Error fetching data from the database: {e}")
        outsiders = []
        non_id_holders = []

    # Render the template with categorized data
    return render(request, 'detection.html', {
        'outsiders': outsiders,
        'non_id_holders': non_id_holders
    })

def camera_id(request):
    return render(request, 'camera_id.html')


DATA_FILE_PATH = os.path.join(settings.BASE_DIR, 'data.json')

def load_data():
    if os.path.exists(DATA_FILE_PATH):
        try:
            with open(DATA_FILE_PATH, 'r') as file:
                data = json.load(file)  # Load the JSON data
                print(data)            # Debug: Print loaded data
                return data            # Return loaded data
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")
            return []
    else:
        print("Error: File does not exist.")
        return []


# Save data to JSON file
def save_data(data):
    with open(DATA_FILE_PATH, 'w') as file:
        json.dump(data, file, indent=4)

# Endpoint to retrieve data
def get_data(request):
    data = load_data()
    return JsonResponse(data, safe=False)

# Endpoint to save new data
@csrf_exempt
def save_data_view(request):
    if request.method == 'POST':
        new_entry = json.loads(request.body)
        data = load_data()
        data.append(new_entry)
        save_data(data)
        return JsonResponse({"message": "Data saved successfully"})

# Endpoint to delete data by index
@csrf_exempt
def delete_data(request, index):
    data = load_data()
    if 0 <= index < len(data):
        data.pop(index)
        save_data(data)
        return JsonResponse({"message": "Data deleted successfully"})
    return JsonResponse({"error": "Index out of range"}, status=400)

def get_camera_ip(camera_id):
    """
    Get the camera IP for the given camera_id from the JSON data.
    """
    camera_data = load_data()
    if 0 <= camera_id < len(camera_data):
        return camera_data[camera_id].get("camera_ip")
    return None


# def generate_video(camera_ip):
#     """
#     Generate video frames from the camera stream identified by camera_id.
#     """
#     # Fetch the camera IP from the JSON file
#     # camera_ip = get_camera_ip(camera_id)
#
#     if not camera_ip:
#         print(f"Error: No camera found for camera_id {camera_id}.")
#         return
#
#     try:
#         # Access the camera stream
#         response = requests.get(f'{camera_ip}', stream=True)
#
#
#         if response.status_code == 200:
#             for chunk in response.iter_content(chunk_size=1024):
#                 yield chunk
#         else:
#             print(f"Error: Failed to access video stream for IP {camera_ip}.")
#     except requests.RequestException as e:
#         print(f"Request failed for camera IP {camera_ip}: {e}")

# def video_feed(request, camera_id=0):
#     """
#     Django view to provide a live video stream.
#     """
#     ips = load_data()
#     camera_ip = ips[camera_id]['camera_ip']
#     return StreamingHttpResponse(
#             generate_video(camera_ip),
#             content_type='multipart/x-mixed-replace; boundary=frame'
#         )

# import cv2
# from django.http import StreamingHttpResponse


# def generate_video(camera_id):
#     """
#     Generate video frames from a hardware camera identified by camera_id.
#     """
#     # Open the video stream using the camera ID (device index)
#     cap = cv2.VideoCapture(camera_id)

#     if not cap.isOpened():
#         print(f"Error: Unable to access camera with ID {camera_id}.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Error: Failed to read frame from camera ID {camera_id}.")
#             break

#         # Encode the frame as JPEG
#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret:
#             print("Error: Failed to encode frame.")
#             break

#         # Yield the frame as a byte stream
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

#     cap.release()


# def video_feed(request, camera_id=0):
#     """
#     Django view to provide a live video stream.
#     """
#     print("Video feed ", camera_id)
#     return StreamingHttpResponse(
#         generate_video(camera_id),
#         content_type='multipart/x-mixed-replace; boundary=frame'
#     )





from django.http import StreamingHttpResponse, HttpResponse
import cv2

def generate_frames(rtsp_url):
    """
    A generator to fetch frames from the RTSP stream.
    """
    cap = cv2.VideoCapture(rtsp_url)  # Open the RTSP stream
    if not cap.isOpened():
        print("Error: Could not open video.")
        return


    while True:
        success, frame = cap.read()

        if not success:

            break  # Exit the loop if no more frames are available

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the proper format for MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def live_stream(request, idx = 0):
    """
    Django view to stream the live video from an RTSP source.
    """
    data = load_data()
    rtsp_url = data[idx]["camera_ip"]
    try:
        # rtsp_url = "rtsp://aiml:Siet@2727@192.168.3.143:554/Streaming/Channels/101"  # Replace with your actual RTSP URL
        return StreamingHttpResponse(
            generate_frames(rtsp_url),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        return HttpResponse("An error occurred while streaming the video.", status=500)

