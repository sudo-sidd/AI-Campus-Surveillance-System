# web/mongo_connection.py
from pymongo import MongoClient
from django.conf import settings

def get_mongo_client():
    client = MongoClient(
        host='mongodb+srv://saidhinakar:PTDB123@ptdb.zlep9.mongodb.net/?retryWrites=true&w=majority&appName=PTDB'
    )
    return client

def get_database():
    client = get_mongo_client()
    return client['DetectionDB']

