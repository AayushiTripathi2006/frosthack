from ultralytics import YOLO
import os

class PlasticDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        # Load YOLO model
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        results = self.model.predict(frame, conf=0.3)
        detections = []

        for result in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, result[:4])
            detections.append((x1, y1, x2, y2))
        return detections
