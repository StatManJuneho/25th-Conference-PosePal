import os
import csv
import pandas as pd
import mediapipe as mp
import cv2
import re

file_name = "wrong"
# Paths
input_dir = f"test/{file_name}/{file_name}_frames"  # Directory containing raw images
output_csv = f"test/{file_name}/{file_name}_pe_timeseries.csv"  # Output CSV for time-series landmarks
output_pose_images_dir = f"test/{file_name}/{file_name}_pe/"  # Directory for pose-annotated images

# Create output directory if it doesn't exist
os.makedirs(output_pose_images_dir, exist_ok=True)

# Mediapipe Pose initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Selected landmarks
selected_landmarks = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_PINKY,
    mp_pose.PoseLandmark.RIGHT_PINKY,
    mp_pose.PoseLandmark.LEFT_INDEX,
    mp_pose.PoseLandmark.RIGHT_INDEX,
    mp_pose.PoseLandmark.LEFT_THUMB,
    mp_pose.PoseLandmark.RIGHT_THUMB,
]

# Initialize CSV columns
landmarks = ['frame_no']  # Add frame number as the primary identifier
for landmark in selected_landmarks:
    landmarks += [f'{landmark.name}_x', f'{landmark.name}_y', f'{landmark.name}_z']

# Create CSV file and write the header
with open(output_csv, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(landmarks)

# Extract frame number from file name
def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)', filename)
    return int(match.group(1)) if match else None

# Save pose landmarks to CSV
def save_pose_landmarks(frame_no, results):
    try:
        keypoints = [frame_no]
        for landmark_id in selected_landmarks:
            landmark = results.pose_landmarks.landmark[landmark_id]
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        # Append data to the CSV
        with open(output_csv, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(keypoints)
    except Exception as e:
        print(f"Error saving landmarks for frame {frame_no}: {e}")

# Process images and save pose landmarks
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_files = [file for file in os.listdir(input_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    frame_files = sorted(frame_files, key=lambda x: extract_frame_number(x))  # Sort frames by NO

    for file in frame_files:
        frame_no = extract_frame_number(file)
        if frame_no is None:
            print(f"Skipping file with invalid frame format: {file}")
            continue

        file_path = os.path.join(input_dir, file)

        # Read and process the image
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Save landmarks to CSV
            save_pose_landmarks(frame_no, results)

            # Draw pose on the image
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            # Save the pose-annotated image
            output_image_path = os.path.join(output_pose_images_dir, file)
            cv2.imwrite(output_image_path, annotated_image)

        print(f"Processed: Frame {frame_no}")

# Load the CSV and structure the data as time series
data = pd.read_csv(output_csv)
data = data.sort_values(by=['frame_no']).reset_index(drop=True)  # Ensure sorting by frame_no

# Save the sorted time-series data
data.to_csv(output_csv, index=False)

print(f"Pose estimation and time-series data saved successfully to {output_csv}.")
