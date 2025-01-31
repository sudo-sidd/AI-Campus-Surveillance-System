# web/mongo_connection.py
from pymongo import MongoClient
from django.conf import settings

def get_mongo_client():
    client = MongoClient(
        host = 'mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/'
        )  # Replace with your MongoDB URI if different

    return client

def get_database():
    client = get_mongo_client()
    # collection = db['DatabaseDB']
    return client['ML_project']