import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        """
        Initialize a tracker using an initial bounding box.
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)  # State transition matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.P *= 10.  # Initial uncertainty
        self.kf.R *= 0.01  # Measurement uncertainty

        self.kf.x[:4] = np.array(bbox).reshape((4, 1))  # Initial state
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, bbox):
        """
        Update the state vector with the new bbox.
        """
        self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box.
        """
        self.kf.predict()
        return self.kf.x[:4].reshape((4,))


class Sort:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        """
        Updates the tracker with new bounding box detections.
        """
        updated_tracks = []
        for detection in detections:
            # Simple logic to assign trackers (for educational purposes)
            bbox = detection[:4]
            tracker = KalmanBoxTracker(bbox)
            updated_tracks.append(np.append(bbox, tracker.id))

        self.trackers = updated_tracks
        return np.array(updated_tracks)
