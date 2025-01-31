from django.http import JsonResponse
from django.shortcuts import render
from .mongo_connection import get_database
from bson import ObjectId
from django.http import HttpResponse, StreamingHttpResponse
import json
import os
from django.views.decorators.csrf import csrf_exempt
from Detection.settings import STATIC_ROOT

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
    time = datetime.now().strftime("%I:%M %p")

    # Generate random _id and get current time
    document = {
        "_id": ObjectId(),  # Generate a random ObjectId
        "Reg_no": reg_no,
        "location": location,
        "time": time,  # Use current UTC time
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
            # role = document.get('role', '').lower()  # Ensure role field is processed in lowercase

            # Categorize by role and ID card status
            if document['Role'] == 'Unidentified':
                outsiders.append(document)
            if (document["Wearing_id_card"] == 'false' or document['Wearing_id_card'] == 'False' or document['Wearing_id_card'] == False) and document['Role']!='Unidentified':  # Check ID card status
                non_id_holders.append(document)

        # Debugging (Optional)
        # print(f"Outsiders: {outsiders}\n")
        # print(f"Non-ID Holders: {non_id_holders}\n")
        
    except Exception as e:
        # Handle any database-related errors gracefully
        print(f"Error fetching data from the database: {e}")
        outsiders = []
        non_id_holders = []

    outsiders = outsiders[::-1]
    non_id_holders = non_id_holders[::-1]
    
    # Render the template with categorized data
    return render(request, 'detection.html', {
        'outsiders': outsiders,
        'non_id_holders': non_id_holders
    })

@csrf_exempt  # Use this only if you want to bypass CSRF checks for this view
def delete_all_documents(request):
        db = get_database()
        collection = db['DatabaseDB']  # Replace with your actual collection name
        
        try:
            # Delete all documents in the collection
            result = collection.delete_many({})  # Empty filter to match all documents
            print('Done!', JsonResponse({'status': 'success', 'message': f'{result.deleted_count} documents deleted successfully.'} ))
            return JsonResponse({'status': 'success', 'message': f'{result.deleted_count} documents deleted successfully.'}, status = 200)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def camera_id(request):
    return render(request, 'camera_id.html')


DATA_FILE_PATH = './data.json'

cached_data = None

def load_data():
    global cached_data
    if cached_data is None:  # Load data only once
        if os.path.exists(DATA_FILE_PATH):
            try:
                with open(DATA_FILE_PATH, 'r') as file:
                    cached_data = json.load(file)  # Load the JSON data
                    print(cached_data)            # Debug: Print loaded data
            except json.JSONDecodeError:
                print("Error: Failed to decode JSON from the file.")
                cached_data = []
        else:
            print("Error: File does not exist.")
            cached_data = []
    return cached_data

# Save data to JSON file
def save_data(data):
    with open(DATA_FILE_PATH, 'w') as file:
        json.dump(data, file, indent=4)

# Endpoint to retrieve data
def get_data(request):
    data = load_data()
    # print(DATA_FILE_PATH)
    return JsonResponse(data, safe = False)

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

def camera_fullscreen(request):
    # Get the camera ID from the URL
    camera_id = request.GET.get('camera')
    if camera_id:
        return render(request, 'fullscreen.html', {'camera_id': camera_id})