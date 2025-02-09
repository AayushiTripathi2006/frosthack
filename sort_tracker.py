import numpy as np

class Sort:
    def update(self, detections):
        tracked_objects = []
        for idx, (x1, y1, x2, y2) in enumerate(detections):
            tracked_objects.append((idx, x1, y1, x2, y2))
        return tracked_objects
