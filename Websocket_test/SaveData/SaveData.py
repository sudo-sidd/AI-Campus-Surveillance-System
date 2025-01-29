from pathlib import Path
from datetime import datetime
import cv2
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging
from .FaceEmbeddings import FaceEmbeddingSystem
from Detection.Detection.settings import STATIC_ROOT


IMAGE_PATH = os.path.join(STATIC_ROOT, 'images')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  SaveData.py
class DataManager:
    def save_data(self, image, context):
        try:
            # Extract face embeddings
            face_embeddings = self.face_system.get_face_embedding(image)
            if not face_embeddings:
                logger.warning("No face detected")
                return None

            # Check for existing person
            existing_person_id = self.face_system.find_matching_person(face_embeddings)

            if existing_person_id:
                # Update existing person's record
                logger.info(f"Updating existing person: {existing_person_id}")

                update_data = {
                    "timestamp": datetime.now(),
                    "camera_location": context['camera_location'],
                    "id_flag": context['id_flag'],
                    "bbox": context['bbox'],
                    "track_id": context['track_id'],
                    "face_flag": context['face_flag'],
                    "face_box": context['face_box'],
                    "id_card": context['id_card'],
                    "id_box": context['id_box'],
                }

                # Update the most recent entry for this person
                self.collection.update_one(
                    {"person_id": existing_person_id},
                    {"$set": update_data},
                    upsert=False  # Don't create new doc if missing
                )
                return existing_person_id

            else:
                # Save new person with embeddings
                logger.info("Saving new person")
                image_name = f"person_{context['camera_location']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = os.path.join(self.image_folder, image_name)

                # Save image
                cv2.imwrite(image_path, image)

                # Create new document
                new_person_id = ObjectId()
                document = {
                    "_id": new_person_id,
                    "person_id": new_person_id,  # Link to self
                    "image_path": f"images/{image_name}",
                    "timestamp": datetime.now(),
                    **context  # Include all context fields
                }

                # Insert into DB and save embeddings
                self.collection.insert_one(document)
                self.face_system._save_embeddings({
                    "doc_id": new_person_id,
                    "embedding": face_embeddings
                })
                return new_person_id

        except Exception as e:
            logger.error(f"Error in save_data: {e}")
            return None