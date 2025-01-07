import argparse
import os
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the face detector
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# Initialize the face recognizer
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)

# Define augmentation transformations
data_augmentation = transforms.Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
])

@torch.no_grad()
def get_feature(face_image):
    # Preprocessing for model input
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Convert to RGB and preprocess
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Get facial features
    emb_img_face = recognizer(face_image)[0].cpu().numpy()
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)  # Normalize
    return images_emb

def process_person_images(person_image_path, name_person, faces_save_dir, images_name, images_emb):
    person_face_path = os.path.join(faces_save_dir, name_person)
    os.makedirs(person_face_path, exist_ok=True)

    for image_name in os.listdir(person_image_path):
        if image_name.lower().endswith(("png", "jpg", "jpeg")):
            input_image = cv2.imread(os.path.join(person_image_path, image_name))
            if input_image is None or input_image.size == 0:
                print(f"Warning: Could not read image {image_name}. Skipping.")
                continue

            try:
                # Detect faces and landmarks
                bboxes, landmarks = detector.detect(image=input_image)

                # Ensure at least one face is detected
                if len(bboxes) == 0:
                    print(f"Warning: No faces detected in {image_name}. Skipping.")
                    continue

                # Process each detected face
                for i in range(len(bboxes)):
                    x1, y1, x2, y2, score = bboxes[i]
                    face_image = input_image[int(y1):int(y2), int(x1):int(x2)]

                    # Skip empty or invalid face images
                    if face_image.size == 0 or face_image.shape[0] < 100 or face_image.shape[1] < 100:
                        print(f"Warning: Invalid face image for {image_name}. Skipping.")
                        continue

                    # Convert face_image to PIL Image
                    face_image_pil = Image.fromarray(face_image)

                    # Apply data augmentation
                    augmented_face_image = data_augmentation(face_image_pil)

                    # Save the original and augmented images
                    for img in [face_image, np.array(augmented_face_image)]:  # Convert augmented image back to NumPy array for saving
                        path_save_face = os.path.join(person_face_path, f"{len(os.listdir(person_face_path))}.jpg")
                        cv2.imwrite(path_save_face, img)
                        images_emb.append(get_feature(face_image=img))
                        images_name.append(name_person)

            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue

def add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path):
    images_name = []
    images_emb = []

    has_subdirs = any(os.path.isdir(os.path.join(add_persons_dir, d)) for d in os.listdir(add_persons_dir))

    if has_subdirs:
        for name_person in os.listdir(add_persons_dir):
            person_image_path = os.path.join(add_persons_dir, name_person)
            if not os.path.isdir(person_image_path):
                continue
            process_person_images(person_image_path, name_person, faces_save_dir, images_name, images_emb)
    else:
        name_person = os.path.basename(add_persons_dir)
        process_person_images(add_persons_dir, name_person, faces_save_dir, images_name, images_emb)

    if not images_emb and not images_name:
        print("No new model found!")
        return None

    images_emb = np.array(images_emb)
    images_name = np.array(images_name)

    features = read_features(features_path)

    if features is not None:
        old_images_name, old_images_emb = features
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))

        print("Update features!")

    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

    if has_subdirs:
        for sub_dir in os.listdir(add_persons_dir):
            dir_to_move = os.path.join(add_persons_dir, sub_dir)
            if os.path.isdir(dir_to_move):
                shutil.move(dir_to_move, backup_dir)
    else:
        backup_person_dir = os.path.join(backup_dir, os.path.basename(add_persons_dir))
        os.makedirs(backup_person_dir, exist_ok=True)
        for file in os.listdir(add_persons_dir):
            shutil.move(os.path.join(add_persons_dir, file), backup_person_dir)

    print("Successfully added new model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add new persons to the face recognition database.")
    parser.add_argument("--backup-dir", type=str, default="./datasets/backup", help="Directory to save model data.")
    parser.add_argument("--add-persons-dir", type=str, default="./datasets/new_persons", help="Directory to add new persons.")
    parser.add_argument("--faces-save-dir", type=str, default="./datasets/data/", help="Directory to save faces.")
    parser.add_argument("--features-path", type=str, default="./datasets/face_features/feature", help="Path to save face features.")
    opt = parser.parse_args()

    for dir_path in [opt.backup_dir, opt.add_persons_dir, opt.faces_save_dir, os.path.dirname(opt.features_path)]:
        os.makedirs(dir_path, exist_ok=True)

    try:
        add_persons(**vars(opt))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
