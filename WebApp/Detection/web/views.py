from django.http import JsonResponse
from django.shortcuts import render
from .mongo_connection import get_database
from bson import ObjectId

def my_view(request):
    db = get_database()
    collection = db['web_person']  # Replace with your actual collection name
    data = list(collection.find())  # Fetch data from the collection

    # Convert ObjectId to string
    for document in data:
        document['_id'] = str(document['_id'])

    return JsonResponse(data, safe=False)  # Return the modified data as a JSON response

def insert_document(request):
    db = get_database()
    collection = db['web_person']
    document = {"name": "example", "value": 42}  # Replace with your actual document
    collection.insert_one(document)
    return JsonResponse({"status": "success"})

def home(request):
    return render(request, 'home.html')

def detection_view(request):
    db = get_database()
    collection = db['web_person']  # Replace with your actual collection name
    data = list(collection.find())

    # Example categorization based on role
    outsiders = []
    non_id_holders = []

    for document in data:
        document['_id'] = str(document['_id'])  # Convert ObjectId to string
        role = document.get('role', '').lower()  # Assuming 'role' field exists
        if role == 'outsider':
            outsiders.append(document)
        else:
            non_id_holders.append(document)

    # print(outsiders)

    return render(request, 'detection.html', {
        'outsiders': outsiders,
        'non_id_holders': non_id_holders
    })
