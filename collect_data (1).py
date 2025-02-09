import cv2
import os

def save_frames_from_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Saved {frame_count} frames to {output_folder}")

# Example usage
save_frames_from_video("../datasets/raw_footage/sample_video.mp4", "../datasets/raw_footage")
