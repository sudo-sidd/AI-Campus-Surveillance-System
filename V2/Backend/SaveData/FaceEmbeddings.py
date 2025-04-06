import face_recognition
import json
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class FaceEmbeddingSystem:
    def __init__(self, json_path='face_embeddings.json'):
        self.json_path = Path(json_path)
        self.embeddings_data = self._load_embeddings()
        self.similarity_threshold = 0.3

        if not self.json_path.exists():
            self._save_embeddings()

    def _load_embeddings(self):
        try:
            if self.json_path.exists():
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                    for person_id in data:
                        data[person_id]['embeddings'] = [
                            np.array(emb) for emb in data[person_id]['embeddings']
                        ]
                    return data
            return {}
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return {}

    def _save_embeddings(self, context=None):
        try:
            match_id = None  # Initialize match_id

            if context:
                match_id = self.find_matching_person(context["embedding"])

                if match_id:
                    # Update existing person's embeddings only
                    logger.info(f"Updating embeddings for existing person ID: {match_id}")
                    self.embeddings_data[match_id]['embeddings'].append(context["embedding"])
                    self.embeddings_data[match_id]['last_updated'] = datetime.now().isoformat()

                else:
                    # Create new person entry with new doc_id
                    person_id = str(len(self.embeddings_data) + 1)
                    logger.info(f"Creating new person with ID: {person_id}")
                    self.embeddings_data[person_id] = {
                        'person_id': person_id,
                        'doc_ids': [str(context["doc_id"])] if "doc_id" in context else [],
                        'embeddings': [context["embedding"]],
                        'last_updated': datetime.now().isoformat(),
                    }

            # Save to JSON
            save_data = {}
            for person_id, person_data in self.embeddings_data.items():
                save_data[person_id] = {
                    'person_id': person_data['person_id'],
                    'doc_ids': person_data.get('doc_ids', []),
                    'embeddings': [emb.tolist() for emb in person_data['embeddings']],
                    'last_updated': person_data['last_updated'],
                }

            self.json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.json_path, 'w') as f:
                json.dump(save_data, f, indent=4)

            logger.info("Embeddings saved successfully")

            # Only return match_id if context was provided
            if context:
                return match_id
            return None

        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise

    def get_face_embedding(self, image):
        try:
            if isinstance(image, str):
                image = cv2.imread(image)
                if image is None:
                    raise ValueError("Could not read image file")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                logger.warning("No face detected in the image")
                return None

            face_encodings = face_recognition.face_encodings(image, face_locations)

            if not face_encodings:
                logger.warning("Could not compute face embedding")
                return None

            return face_encodings[0]

        except Exception as e:
            logger.error(f"Error getting face embedding: {e}")
            return None

    def find_matching_person(self, embedding):
        if embedding is None:
            return None

        try:
            best_match = None
            highest_similarity = 0

            for person_id, person_data in self.embeddings_data.items():
                for stored_embedding in person_data['embeddings']:
                    face_distance = face_recognition.face_distance([stored_embedding], embedding)[0]
                    similarity = 1 - face_distance

                    if similarity > self.similarity_threshold and similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = person_id

            return best_match

        except Exception as e:
            logger.error(f"Error finding matching person: {e}")
            return None