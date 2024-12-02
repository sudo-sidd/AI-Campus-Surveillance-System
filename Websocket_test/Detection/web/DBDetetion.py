import pymongo
from typing import Union, List, Dict, Optional
import json

class MongoDBDeleter:
    def __init__(self):
        connection_string = 'mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/'
        database_name = 'ML_project'
        try:
            # Create a MongoDB client
            self.client = pymongo.MongoClient(connection_string)
            
            # Select the database
            self.db = self.client[database_name]
            self.collection = self.db['DatabaseDB'] 
            
            print(f"Connected to database: {database_name}")
        except pymongo.errors.ConnectionFailure as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise

    def delete_all_documents(self) -> int:
        """
        Delete all documents in a specific collection
        :return: Number of deleted documents
        """
        
        try:
            result = self.db[self.collection].delete_many({})
            print(f"Deleted {result.deleted_count} documents from {self.collection}")
            return result.deleted_count
        except Exception as e:
            print(f"Error deleting documents from {self.collection}: {e}")
            return 0


    def list_collections(self) -> List[str]:
        """
        List all collections in the database
        
        :return: List of collection names
        """
        try:
            collections = self.db.list_collection_names()
            print("Available collections:", collections)
            return collections
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []

    def close_connection(self):
        """
        Close the MongoDB connection
        """
        try:
            self.client.close()
            print("MongoDB connection closed")
        except Exception as e:
            print(f"Error closing connection: {e}")

    def list_documents(self):
        """
        View to categorize and display individuals based on their role and ID card status.
        """
        # Connect to the database
        db = self.db
        collection = db['DatabaseDB']  # Replace with your actual collection name
        
        try:
            # Fetch all documents from the collection
            data = list(collection.find())
            
            print("Data : ")
            for document in data:
                document['_id'] = str(document['_id']) 
                print(document)
        except Exception as e:
            print("Error : ",e)
