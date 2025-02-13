from datetime import datetime
from pathlib import Path
import logging
from pymongo import MongoClient
from .FaceEmbeddings import FaceEmbeddingSystem
import cv2
from bson.objectid import ObjectId
from Detection.Detection.settings import STATIC_ROOT
import os

IMAGE_PATH = os.path.join(STATIC_ROOT, 'images')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SaveData.py
class DataManager:
    def __init__(self, mongo_uri, db_name, collection_name, image_folder=IMAGE_PATH):
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

    def _generate_unique_image_name(self, base_name):
        """Generate a unique image name by appending a counter if necessary"""
        counter = 1
        image_path = self.image_folder / base_name

        while image_path.exists():
            name_parts = base_name.rsplit('.', 1)
            new_name = f"{name_parts[0]}_{counter}.{name_parts[1]}"
            image_path = self.image_folder / new_name
            counter += 1

        return image_path

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
                # For existing person, only update embeddings
                logger.info(f"Face matches existing person ID: {existing_face}")
                self.face_system._save_embeddings({
                    'embedding': face_embeddings
                })
                return existing_face  # Return existing person ID

            else:
                # For new person, create document and save embeddings
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_image_name = f"person_{context['camera_location']}_{timestamp}.jpg"
                image_path = self._generate_unique_image_name(base_image_name)

                # Save image only for new person
                cv2.imwrite(str(image_path), image)

                # Create MongoDB document for new person
                doc_id = ObjectId()
                document = {
                    "_id": doc_id,
                    'bbox': context['bbox'],
                    'timestamp': timestamp,
                    'track_id': context['track_id'],
                    'face_flag': context['face_flag'],
                    'face_box': context['face_box'],
                    'id_flag': context['id_flag'],
                    'id_card': context['id_card'],
                    'id_box': context['id_box'],
                    'camera_location': context['camera_location'],
                }

                # Save embeddings with new document ID
                self.face_system._save_embeddings({
                    'doc_id': doc_id,
                    'embedding': face_embeddings
                })

                # Save document to MongoDB
                self.collection.insert_one(document)
                logger.info(f"New person document saved with ID: {document['_id']}")
                return doc_id

        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return None