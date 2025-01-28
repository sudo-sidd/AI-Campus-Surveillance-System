from pathlib import Path
from datetime import datetime
import cv2
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging
from .FaceEmbeddings import FaceEmbeddingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  SaveData.py
class DataManager:
    def __init__(self, mongo_uri, db_name, collection_name, image_folder='./images/'):
        self.image_folder = Path(image_folder)
        self.image_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.face_system = FaceEmbeddingSystem()
            logger.info('Database connected successfully')
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            raise

    def save_data(self, image, context):
        try:
            # Get face embeddings first
            face_embeddings = self.face_system.get_face_embedding(image)
            if face_embeddings is None:
                logger.warning("No face detected in the image")
                return None

            # Check if face already exists
            existing_face = self.face_system.find_matching_person(face_embeddings)
            if existing_face:
                logger.info(f"Face matches existing person ID: {existing_face}")
                # Update embeddings for existing person
                doc_id = ObjectId()
                self.face_system._save_embeddings({
                    'doc_id': doc_id,
                    'embedding': face_embeddings
                })
                
                # Save only the new document reference
                document = {
                    "_id": doc_id,
                    "location": context['camera_location'],
                    "timestamp": datetime.now(),
                    "person_status": "known",  # Update status since we found a match
                    "id_card_status": context['id_card_status'],
                    "person_id": existing_face  # Link to existing person
                }
            else:
                # Handle new face
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = f"person_{context['camera_location']}_{timestamp}.jpg"
                image_path = self.image_folder / image_name

                doc_id = ObjectId()
                document = {
                    "_id": doc_id,
                    "location": context['camera_location'],
                    "timestamp": datetime.now(),
                    "person_status": context['person_status'],
                    "id_card_status": context['id_card_status'],
                    "image_path": str(image_path)
                }

                # Save image only for new faces
                cv2.imwrite(str(image_path), image)
                
                # Save embeddings for new person
                self.face_system._save_embeddings({
                    'doc_id': doc_id,
                    'embedding': face_embeddings
                })

            # Save document to MongoDB
            self.collection.insert_one(document)
            logger.info(f"Document saved with ID: {document['_id']}")
            return document['_id']

        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return None

    def check_data(self, image, context):
        """Check if person exists and update embeddings"""
        try:
            result = self.face_system.identify_person(image)
            
            if result and result['matched']:
                # Only update embedding data, don't save new image
                self.face_system._save_embeddings(context)
                logger.info(f"Updated embeddings for existing person: {result['person_id']}")
                return result['person_id']
            else:
                # Save new person data
                self.face_system._save_embeddings(context)
                logger.info("Saved embeddings for new person")
                return None
                
        except Exception as e:
            logger.error(f"Error checking data: {e}")
            raise