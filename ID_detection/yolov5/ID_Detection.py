import torch
import cv2
import numpy as np

model = torch.hub.load('ID_detection/yolov5', 'custom', path='ID_detection/yolov5/best.pt', source='local')

model.eval()

def detect_id_card(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)
    print(results)
    results.render()

    frame_with_detections = np.squeeze(results.ims) 
    frame_with_detections = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)

    return frame_with_detections
