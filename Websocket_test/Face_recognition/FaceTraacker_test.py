from collections import defaultdict, Counter
import time


class FaceTracker:
    def __init__(self, timeout=2.0, detection_window=6, threshold=0.5, update_interval=3.0):
        self.tracker = defaultdict(lambda: {
            "state": "PENDING",  # Initialize state as pending for new faces
            "last_seen": None,
            "box": None,
            "detections": [],  # Store detection results (including names)
            "timestamps": [],  # Store detection timestamps for window management
            "history": []  # Store past detection windows for contradiction resolution
        })
        self.timeout = timeout
        self.detection_window = detection_window  # Define how many frames to collect detections
        self.threshold = threshold
        self.update_interval = update_interval
        self.last_update_time = 0  # Time of the last state update

    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate Intersection over Union (IOU) between two boxes.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def match_face(self, face_box):
        """
        Match a detected face to an existing tracker using IOU.
        """
        best_match = None
        best_iou = 0.0

        for face_id, data in self.tracker.items():
            iou = self.calculate_iou(face_box, data["box"])
            if iou > best_iou and iou > 0.5:  # Match only if IOU > 0.5
                best_match = face_id
                best_iou = iou

        return best_match

    def update_face(self, face_box, detection, current_time):
        """
        Update or add a face to the tracker.
        """
        face_id = self.match_face(face_box)

        if face_id is None:  # No match, add a new face
            face_id = tuple(face_box)
            # Initialize with PENDING state directly
            self.tracker[face_id] = {
                "state": "PENDING",  # Placeholder for unclassified faces
                "last_seen": current_time,
                "box": face_box,
                "detections": [detection],  # Start tracking detections
                "timestamps": [current_time],  # Store timestamps
                "history": [detection],  # Save detection for contradiction resolution
            }
        else:
            # Update the tracker for the matched face
            self.tracker[face_id]["last_seen"] = current_time
            self.tracker[face_id]["box"] = face_box
            self.tracker[face_id]["detections"].append(detection)
            self.tracker[face_id]["timestamps"].append(current_time)

            # If current detection contradicts previous state, store it in history
            if self.tracker[face_id]["state"] != "PENDING" and detection != self.tracker[face_id]["state"]:
                self.tracker[face_id]["history"].append(detection)

        # Prune detections to fit the detection window
        self.prune_old_detections(face_id, current_time)

    def prune_old_detections(self, face_id, current_time):
        """
        Prune old detections that are older than the detection window.
        """
        # Remove detections older than the detection window
        timestamps = self.tracker[face_id]["timestamps"]
        detections = self.tracker[face_id]["detections"]

        while timestamps and timestamps[0] < current_time - self.detection_window:
            timestamps.pop(0)
            detections.pop(0)

    # def update_states(self, current_time):
    #     """
    #     Update face states based on detection history at intervals.
    #     """
    #     if current_time - self.last_update_time < self.update_interval:
    #         return  # Update states only every few seconds
    #
    #     self.last_update_time = current_time  # Update the last update time
    #
    #     for face_id, data in list(self.tracker.items()):
    #         # Remove faces not seen for too long
    #         if current_time - data["last_seen"] > self.timeout:
    #             del self.tracker[face_id]
    #             continue
    #
    #         # Evaluate the detection history and select the most frequent result after multiple detections
    #         if len(data["history"]) >= 3:
    #             # Evaluate the detection history and select the most frequent result
    #             count = Counter(data["history"])
    #             most_common_detection, _ = count.most_common(1)[0]
    #
    #             # If the most common detection is different from the current state, update it
    #             if most_common_detection != data["state"]:
    #                 # Update state only if the detection persists over 3 detection windows (9 seconds)
    #                 data["state"] = most_common_detection
    #
    #         # Calculate state based on detection history
    #         total_detections = len(data["detections"])
    #         if total_detections > 0:
    #             # Count the occurrences of each detection result (including names)
    #             count = Counter(data["detections"])
    #             most_common_detection, _ = count.most_common(1)[0]
    #
    #             # If the most common detection is different than the current state, update it
    #             if most_common_detection != data["state"]:
    #                 data["state"] = most_common_detection

    def update_states(self, current_time):
        """
        Update face states based on detection history at intervals.
        """
        if current_time - self.last_update_time < self.update_interval:
            return  # Update states only every few seconds

        self.last_update_time = current_time  # Update the last update time

        for face_id, data in list(self.tracker.items()):
            # Remove faces not seen for too long
            if current_time - data["last_seen"] > self.timeout:
                del self.tracker[face_id]
                continue

            # Resolve contradiction after 3 detection windows (9 seconds)
            if len(data["history"]) >= 3:
                # Evaluate the detection history and select the most frequent result
                count = Counter(data["history"])
                most_common_detection, _ = count.most_common(1)[0]
                data["state"] = most_common_detection

            # Calculate state based on detection history
            total_detections = len(data["detections"])
            if total_detections > 0:
                # Count the occurrences of each detection result (including names)
                count = Counter(data["detections"])
                most_common_detection, _ = count.most_common(1)[0]

                # If the most common detection is different than the current state, update it
                if most_common_detection != data["state"]:
                    data["state"] = most_common_detection

    def get_tracked_faces(self):
        """
        Return the tracked faces with their state and bounding box.
        """
        return [
            {"state": data["state"], "box": data["box"]}
            for data in self.tracker.values()
        ]
