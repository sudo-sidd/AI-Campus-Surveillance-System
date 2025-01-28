import cv2
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(BASE_DIR,"model", "person_detection.pt")

yolo = YOLO(yolo_path)

def draw_track(frame, track):
    bbox = track['bbox']
    track_id = track['id']
    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    text = f'ID: {track_id}'
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_width, y1), (255, 255, 255), -1)
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def track_persons(frame):

    results = yolo.track(frame, persist=True)

    if not results[0].boxes:
        return {"modified_frame": frame, "person_boxes": [], "track_ids": []}

    modified_frame = results[0].plot()
    person_bboxes = results[0].boxes.xywh.cpu().numpy()  # Bounding boxes
    track_ids = results[0].boxes.id.int().cpu().tolist()  # Tracking IDs

    return {
        "modified_frame": modified_frame,
        "person_boxes": person_bboxes,
        "track_ids": track_ids
    }


#
# frame = cv2.imread('ppl1.jpg')
#
# results = track_persons(frame)
#
# annotated_frame = results["modified_frame"]
# bounding_boxes = results["person_boxes"]
# track_ids = results["track_ids"]
#
# cv2.imwrite('op_frame.png', annotated_frame)
# print(bounding_boxes,'\n')
# print(track_ids)
# cv2.waitKey(0)
