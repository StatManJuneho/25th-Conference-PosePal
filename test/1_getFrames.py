import cv2
import os
import sys
import math


video_file_name = "wrong"

video_path = f"test/videos/{video_file_name}.mp4"
output_dir = f"test/{video_file_name}/{video_file_name}_frames"
desired_fps = 30  # number of frames to extract per second

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    sys.exit(1)

original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = math.floor(original_fps / desired_fps) if original_fps > 0 else 1

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save a frame every 'frame_interval' frames
    if frame_count % frame_interval == 0:
        # Construct the filename
        frame_filename = os.path.join(output_dir, f"frame_{frame_count}.png")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Extracted {saved_count} frames to the directory '{output_dir}'.")
