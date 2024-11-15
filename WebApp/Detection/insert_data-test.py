from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timezone

def insert_document():
    # Set up MongoDB client
    client = MongoClient('mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/')  # Replace with your MongoDB URI if different
    db = client['ML_project']  # Replace with your database name
    collection = db['DatabaseDB']  # Replace with your collection name

    # Collect document details from user input
    reg_no = int(input("Enter Reg_no (e.g., 714023202013): "))
    location = input("Enter location (e.g., aiml dept): ")
    role = input("Enter Role (e.g., student): ")
    wearing_id_card = input("Is wearing ID card? (true/false): ").strip().lower() == 'true'
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

    # Insert document
    result = collection.insert_one(document)
    print("Document inserted with _id:", result.inserted_id)

# Call the function
insert_document()