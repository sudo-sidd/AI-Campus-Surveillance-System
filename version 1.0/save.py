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
    x1,x2,y1,y2=[0,0,1,1]
    track_id= 0
    camera_data={
        'camera_location':'main gate'
    }
    
    context = {
                'bbox': [x1, y1, x2, y2],
                'track_id': track_id,
                'face_flag': "UNKNOWN",
                'face_box': [0, 0, 0, 0],
                'id_flag': False,
                'id_card':'none',
                'id_box': [0, 0, 0, 0],
                'camera_location': camera_data['camera_location']
            }



# test_image_path =  '/mnt/sda1/old- Face_rec-ID_detection/version 1.0/test_img2.jpg'
# test_image = cv.imread(test_image_path)


# consider the passing image is marked as unknown in face recognition part itself
# so we can directly pass the image to the function

# save_data(test_image, context)
# result = data_manager.save_data(test_image, context)
# print(result)