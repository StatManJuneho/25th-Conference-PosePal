###
# 5개의 프레임을 뽑는 코드
###
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

file_name = "wrong"
# Set input and output directories
segments_dir = f"test/{file_name}/{file_name}_segments"  # Directory containing segment files
selected_points_dir = f"test/{file_name}/{file_name}_segments"  # Directory to save selected points
os.makedirs(selected_points_dir, exist_ok=True)

# Get list of segment files
segment_files = [f for f in os.listdir(segments_dir) if f.startswith("rep_segment_") and f.endswith(".csv")]

# Iterate through each segment file
for segment_file in segment_files:
    segment_path = os.path.join(segments_dir, segment_file)
    segment_data = pd.read_csv(segment_path)

    # Extract keypoint-coordinate columns
    keypoint_columns = [col for col in segment_data.columns if "_" in col and col != "frame_no"]

    # Select meaningful frames for the entire segment: start, intermediate points, peaks, and end
    start_idx = 0
    end_idx = len(segment_data) - 1

    # Detect peaks within the segment based on the average of all keypoints
    avg_series = segment_data[keypoint_columns].mean(axis=1).values
    peaks, _ = find_peaks(avg_series)

    # Ensure exactly 5 selected rows: start, midpoint between start and peak, peak, midpoint between peak and end, and end
    selected_indices = [start_idx]

    if len(peaks) > 0:
        # Use the first peak as reference
        peak_idx = peaks[0]
        selected_indices.append(peak_idx)

        # Add intermediate points
        mid_start_peak_idx = (start_idx + peak_idx) // 2
        mid_peak_end_idx = (peak_idx + end_idx) // 2

        selected_indices.extend([mid_start_peak_idx, mid_peak_end_idx])
    else:
        # If no peak is found, add midpoint
        midpoint_idx = (start_idx + end_idx) // 2
        selected_indices.append(midpoint_idx)

    selected_indices.append(end_idx)

    # Ensure unique indices and sort them
    selected_indices = sorted(set(selected_indices))

    # Collect the selected rows
    selected_data = segment_data.iloc[selected_indices]

    # Save the selected rows to a new CSV file
    selected_rows_file_name = f"processed_{segment_file}"
    selected_data.to_csv(os.path.join(selected_points_dir, selected_rows_file_name), index=False)

    print(f"Selected rows saved for {segment_file}: {selected_rows_file_name}")
