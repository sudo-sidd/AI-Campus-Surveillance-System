from collections import defaultdict

class FaceTracker:
    def __init__(self, timeout=2.0, detection_window=3, threshold=0.6):
        self.tracker = defaultdict(lambda: {
            "state": "UNDETERMINED",
            "last_seen": None,
            "box": None,
            "detections": [],
        })
        self.timeout = timeout
        self.detection_window = detection_window
        self.threshold = threshold

    def update_face(self, face_id, face_box, detection, current_time):
        # Update or initialize face tracking info
        if face_id not in self.tracker:
            self.tracker[face_id] = {"state": "UNDETERMINED", "last_seen": current_time, "box": face_box,
                                     "detections": []}

        self.tracker[face_id]["last_seen"] = current_time
        self.tracker[face_id]["box"] = face_box
        self.tracker[face_id]["detections"].append(detection)

        # Prune detections older than the detection window
        if len(self.tracker[face_id]["detections"]) > self.detection_window:
            self.tracker[face_id]["detections"].pop(0)

    def update_states(self, current_time):
        # Update states based on detection history
        for face_id, data in list(self.tracker.items()):
            if current_time - data["last_seen"] > self.timeout:
                # Forget face if not seen for too long
                del self.tracker[face_id]
                continue

            total_detections = len(data["detections"])
            if total_detections >= self.detection_window:
                sietian_count = sum(1 for d in data["detections"] if "SIETIAN" in d)
                sietian_ratio = sietian_count / total_detections
                if sietian_ratio > self.threshold:
                    data["state"] = "SIETIAN"
                elif sietian_ratio < (1 - self.threshold):
                    data["state"] = "UNKNOWN"
                else:
                    data["state"] = "UNDETERMINED"

    def get_state(self, face_id):
        # Get state for a specific face
        return self.tracker.get(face_id, {}).get("state", "UNDETERMINED")

    def prune_lost_faces(self, current_time):
        # Remove faces not seen within the timeout
        lost_faces = [face_id for face_id, data in self.tracker.items() if
                      current_time - data["last_seen"] > self.timeout]
        for face_id in lost_faces:
            del self.tracker[face_id]
