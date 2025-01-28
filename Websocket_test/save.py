# # def save_detections(people_data, camera_location):
# #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     for person in people_data:
# #         image_name = f"person_{camera_location}_{current_time}.jpg"
# #         image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)

# #         os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)
# #         cv2.imwrite(image_path, person['bbox'])

# #         existing_document = collection.find_one({"track_id": person['track_id']})

# #         if existing_document:
# #             # If the document exists, delete the old entry
# #             collection.delete_one({"_id": existing_document["_id"]})
# #             print(f"Deleted old entry with _id: {existing_document['_id']}")

# #         document = {
# #             "_id": ObjectId(),
# #             "location": camera_location,
# #             "time": datetime.now().strftime("%D %I:%M %p"),
# #             "Role": "Unidentified" if person['face_flag'] == "UNKNOWN" else "SIETIAN",
# #             "Wearing_id_card": person['id_card_status'],
# #             "image": f"person_{camera_location}_{current_time}.jpg",
# #             "track_id": person['track_id']
# #         }
# #         result = collection.insert_one(document)
# #         print(f"Document inserted with _id: {result.inserted_id}")




# import face_recognition
# import json
# import numpy as np
# import cv2
# from datetime import datetime
# from pathlib import Path

# class FaceEmbeddingSystem:
#     def __init__(self, json_path='face_embeddings.json'):
#         self.json_path = Path(json_path)
#         self.embeddings_data = self._load_embeddings()
#         self.similarity_threshold = 0.6  # Adjust this threshold based on your needs
        
#     def _load_embeddings(self):
#         """Load existing embeddings from JSON file or create new if doesn't exist"""
#         if self.json_path.exists():
#             with open(self.json_path, 'r') as f:
#                 data = json.load(f)
#                 # Convert string embeddings back to numpy arrays
#                 for person_id in data:
#                     data[person_id]['embeddings'] = [
#                         np.array(emb) for emb in data[person_id]['embeddings']
#                     ]
#                 return data
#         return {}

#     def _save_embeddings(self):
#         """Save embeddings to JSON file"""
#         # Convert numpy arrays to lists for JSON serialization
#         save_data = {}
#         for person_id, person_data in self.embeddings_data.items():
#             save_data[person_id] = {
#                 'name': person_data['name'],
#                 'embeddings': [emb.tolist() for emb in person_data['embeddings']],
#                 'last_updated': person_data['last_updated']
#             }
        
#         with open(self.json_path, 'w') as f:
#             json.dump(save_data, f, indent=4)

#     def get_face_embedding(self, image_path):
#         """Extract face embedding from an image"""
#         # Load image
#         image = face_recognition.load_image_file(image_path)
        
#         # Find face locations
#         face_locations = face_recognition.face_locations(image)
        
#         if not face_locations:
#             raise ValueError("No face detected in the image")
        
#         # Get face encodings
#         face_encodings = face_recognition.face_encodings(image, face_locations)
        
#         if not face_encodings:
#             raise ValueError("Could not compute face embedding")
            
#         return face_encodings[0]

#     def add_person(self, image_path, name):
#         """Add a new person or update existing person's embeddings"""
#         try:
#             embedding = self.get_face_embedding(image_path)
            
#             # Check if this face matches any existing person
#             match_id = self.find_matching_person(embedding)
            
#             if match_id:
#                 # Add to existing person's embeddings
#                 self.embeddings_data[match_id]['embeddings'].append(embedding)
#                 self.embeddings_data[match_id]['last_updated'] = datetime.now().isoformat()
#                 print(f"Added new embedding to existing person: {self.embeddings_data[match_id]['name']}")
#             else:
#                 # Create new person entry
#                 person_id = str(len(self.embeddings_data) + 1)
#                 self.embeddings_data[person_id] = {
#                     'name': name,
#                     'embeddings': [embedding],
#                     'last_updated': datetime.now().isoformat()
#                 }
#                 print(f"Added new person: {name}")
            
#             self._save_embeddings()
#             return True
            
#         except Exception as e:
#             print(f"Error adding person: {str(e)}")
#             return False

#     def find_matching_person(self, embedding):
#         """Find if the given embedding matches any existing person"""
#         for person_id, person_data in self.embeddings_data.items():
#             for stored_embedding in person_data['embeddings']:
#                 # Calculate face distance
#                 face_distance = face_recognition.face_distance([stored_embedding], embedding)[0]
                
#                 # Convert distance to similarity score (lower distance = higher similarity)
#                 similarity = 1 - face_distance
                
#                 if similarity > self.similarity_threshold:
#                     return person_id
#         return None

#     def identify_person(self, image_path):
#         """Identify person from image"""
#         try:
#             embedding = self.get_face_embedding(image_path)
#             match_id = self.find_matching_person(embedding)
            
#             if match_id:
#                 person_data = self.embeddings_data[match_id]
#                 return {
#                     'matched': True,
#                     'person_id': match_id,
#                     'name': person_data['name']
#                 }
#             else:
#                 return {
#                     'matched': False,
#                     'person_id': None,
#                     'name': None
#                 }
                
#         except Exception as e:
#             print(f"Error identifying person: {str(e)}")
#             return None

# # Example usage
# if __name__ == "__main__":
#     # Initialize the system
#     face_system = FaceEmbeddingSystem()
    
#     # Add a new person
#     # face_system.add_person("./image/image copy.png", "Iron Man")
#     # face_system.add_person("./image/image copy 2.png", "Iron Man")
    
#     # Add another image of the same person (will be automatically grouped)
#     # face_system.add_person("./image/thoughtful-woman-with-hand-on-chin-looking-up.webp", "Unknown")
    
#     # Try to identify a person
#     result = face_system.identify_person("./image/ppl1.jpg")
#     if result:
#         if result['matched']:
#             print(f"Person identified as: {result['name']}")
#         else:
#             print("No matching person found")


from SaveData.SaveData import DataManager
import os
import cv2 as cv

if __name__ == "__main__":
    # Initialize DataManager
    data_manager = DataManager(
        mongo_uri=os.getenv("MONGO_URI", "mongodb+srv://ml_dept_project:ml_dept_project@ml-project.gkigx.mongodb.net/"),
        db_name='ML_project',
        collection_name='DatabaseDB'
    )

    # Example context
    context = {
        "camera_location": "entrance",
        "person_status": "unknown",
        "id_card_status": "not_wearing"
    }



test_image_path =  '/mnt/sda1/Face_rec-ID_detection/New-work/FaceRecognizer/image/thoughtful-woman-with-hand-on-chin-looking-up.webp'
test_image = cv.imread(test_image_path)


# consider the passing image is marked as unknown in face recognition part itself
# so we can directly pass the image to the function

# save_data(test_image, context)
result = data_manager.save_data(test_image, context)
print(result)