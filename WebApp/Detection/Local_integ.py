import os
import django
import cv2
import time
from datetime import datetime
from django.core.files.base import ContentFile
from Face_recognition.face_recognize import recognize_face
from ID_detection.yolov11.ID_Detection import detect_id_card
from web.models import Person

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CCTV.settings')
django.setup()


def process_and_save_detections(frame, person_boxes, flags, associations, camera_id):
    """
    Process detections and save to database

    Args:
        frame: The video frame
        person_boxes: List of person bounding boxes
        flags: List of recognition flags ("SIETIAN", "UNKNOWN", or "_")
        associations: ID card associations from detect_id_card
        camera_id: Identifier for the camera
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, (person_box, flag) in enumerate(zip(person_boxes, flags)):
        if flag == "_":  # Skip if no face detected
            continue

        # Extract person sub-image from frame
        x1, y1, x2, y2 = [int(coord) for coord in person_box]
        person_image = frame[y1:y2, x1:x2]

        # Get ID card type from associations
        id_card_type = associations.get(idx, None)  # Assuming associations maps person index to ID type
        wearing_id_card = bool(id_card_type)

        # Generate unique image name
        image_name = f"person_{camera_id}_{current_time}_{idx}.jpg"

        # Prepare data for database based on recognition flags
        if flag.startswith("SIETIAN"):  # Known face recognized as SIETIAN
            person_name = flag.split(" ")[0]  # Extract name if needed (e.g., "SIETIAN (John Doe)")
            roll = "Student"  # Or get from your recognition system
            recognition_status = "Recognized"
        else:  # UNKNOWN or other statuses
            person_name = "Unknown Person"
            roll = "Unidentified"
            recognition_status = "Unknown"

        # Save to database only if the face is unknown or recognized without ID card
        if recognition_status == "Unknown" or (recognition_status == "Recognized" and not wearing_id_card):
            save_person_record(
                frame=person_image,
                person_name=person_name,
                role=roll,
                wearing_id_card=wearing_id_card,
                id_card_type=id_card_type,
                location=camera_id,
                image_name=image_name,
                recognition_status=recognition_status
            )


def save_person_record(frame, person_name, role, wearing_id_card, id_card_type,
                       location, image_name, recognition_status):
    """
    Save a Person record to the database.
    """
    # Convert frame to an image file and save it using Django's File system.
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_file = ContentFile(img_encoded.tobytes(), name=image_name)

    # Create a Person record and save it to the database.
    person_record = Person(
        name=person_name,
        role=role,
        wearing_id_card=wearing_id_card,
        id_card_type=id_card_type,
        location=location,
        recognition_status=recognition_status
    )

    person_record.image.save(image_name, img_file)
    person_record.save()


def video_feed(camera_id=0):
    """
    Main video processing loop.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize time tracking for periodic saving.
    last_save_time = time.time()
    save_interval = 2.0  # Save every 2 seconds

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image.")
            break

        # Get person boxes and ID card detections.
        modified_frame, person_boxes, associations = detect_id_card(frame)

        # Get face recognition results.
        modified_frame, flags = recognize_face(modified_frame, person_boxes)

        # Save detections periodically.
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            process_and_save_detections(
                frame=frame,
                person_boxes=person_boxes,
                flags=flags,
                associations=associations,
                camera_id=camera_id
            )
            last_save_time = current_time

        # Display the processed frame.
        cv2.imshow('Video Feed', modified_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_feed()
