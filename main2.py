import os
import cv2
import pandas as pd
from ultralytics import YOLO

# Paths and configurations
VIDEO_PATH = "test_video.mp4"  # Replace with your video path
FRAME_FOLDER = "./frames"
OUTPUT_CSV = "submit.csv"
YOLO_MODEL_PATH = "YOLO_Custom_v8m.pt"

# Create folders if not present
os.makedirs(FRAME_FOLDER, exist_ok=True)

# Load YOLO Model
model = YOLO(YOLO_MODEL_PATH)

# DataFrame for results
df = pd.DataFrame(columns=['Frame', 'Geo_Tag_URL', 'Prediction'])

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
frame_index = 0

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_filename = os.path.join(FRAME_FOLDER, f"frame_{frame_index:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    # Perform YOLO prediction
    results = model.predict(source=frame_filename, save=True, conf=0.37)

    # Count plastic detections
    plastic_count = sum(1 for result in results for label in result.names.values() if "plastic" in label.lower())

    prediction_text = "Waste Plastic Detected" if plastic_count > 10 else "No Waste Plastic"

    # Append frame info to the DataFrame
    df.loc[len(df)] = [f"frame_{frame_index:04d}.jpg", "28.6139°N 77.2090°E", prediction_text]

    frame_index += 1

cap.release()

# Save results to CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"Processing complete. CSV saved as '{OUTPUT_CSV}'.")
