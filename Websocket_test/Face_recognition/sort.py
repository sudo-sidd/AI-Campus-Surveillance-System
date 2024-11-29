import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.bbox = bbox
        self.time_since_update = 0
        self.hits = 0

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        self.time_since_update += 1
        return self.bbox

class Sort:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        updated_tracks = []
        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append(KalmanBoxTracker(det))
        else:
            matches = self._assign_detections_to_trackers(detections)
            for t, d in matches:
                self.trackers[t].update(detections[d])
                updated_tracks.append(self.trackers[t].predict())

        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]
        return np.array(updated_tracks)

    def _assign_detections_to_trackers(self, detections):
        iou_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
        for t, tracker in enumerate(self.trackers):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._iou(tracker.bbox, det)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matches = [(r, c) for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] > self.iou_threshold]
        return matches

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)
