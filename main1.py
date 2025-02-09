import torch
from ultralytics import YOLO  # Ensure ultralytics is installed
from sort import Sort  # SORT tracker for object tracking

# Load the custom YOLO model
model_path = 'YOLO_Custom_v8m.pt'
try:
    model = YOLO(model_path)
    print(f"Loaded YOLO model from: {model_path}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")

# Initialize SORT for tracking
tracker = Sort()

# Dummy detection data (replace this with actual inference results)
detections = [[50, 50, 200, 200, 0.9], [300, 100, 400, 300, 0.8]]  # [x1, y1, x2, y2, confidence]

# Update SORT with detections
tracked_objects = tracker.update(detections)

# Display tracked object information
for obj in tracked_objects:
    if len(obj) == 6:  # Ensure there are six values
        x1, y1, x2, y2, confidence, obj_id = obj
        print(f"Tracked Object ID {int(obj_id)}: Bounding Box [{x1}, {y1}, {x2}, {y2}] with confidence {confidence}")
