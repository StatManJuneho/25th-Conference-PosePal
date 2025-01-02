###
# Peak Detection and Extract Reps
###
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

file_name = "wrong"

# Set input and output directories
input_file = f"test/{file_name}/{file_name}_smoothed_timeseries.csv"  # Path to the smoothed time-series CSV file
output_dir = f"test/{file_name}"  # Directory to save results
os.makedirs(output_dir, exist_ok=True)
output_segments_dir = os.path.join(output_dir, f"{file_name}_segments")
os.makedirs(output_segments_dir, exist_ok=True)

# Load the smoothed CSV file
data = pd.read_csv(input_file)

# Extract the list of keypoint-coordinate columns (e.g., LEFT_ELBOW_y)
keypoint_columns = [col for col in data.columns if "_" in col and col != "frame_no"]

# Function to detect peaks and troughs with enhanced logic
def detect_peaks_and_troughs(smoothed_series):
    # Detect raw peaks and troughs
    raw_peaks, _ = find_peaks(smoothed_series, height=0, distance=5)
    raw_troughs, _ = find_peaks(-smoothed_series, distance=5)

    # Custom detection for near-zero troughs
    zero_threshold = 0.01  # Adjust based on the data range
    zero_troughs = np.where(np.abs(smoothed_series) < zero_threshold)[0]

    # Include the first trough if it exists
    if smoothed_series[0] < smoothed_series[1]:  # Check if the first point is a trough
        zero_troughs = np.append(zero_troughs, 0)

    # Combine and deduplicate trough indices
    all_troughs = sorted(set(raw_troughs).union(zero_troughs))

    # Refine extrema detection
    refined_troughs = []
    for trough in all_troughs:
        if 0 <= trough < len(smoothed_series) - 1:  # Include the first trough explicitly
            if (trough == 0 or smoothed_series[trough] < smoothed_series[trough - 1]) and \
               (trough == len(smoothed_series) - 1 or smoothed_series[trough] < smoothed_series[trough + 1]):
                refined_troughs.append(trough)

    return refined_troughs

# Detect troughs using RIGHT_ELBOW_y as a reference
sample_keypoint = "RIGHT_ELBOW"
sample_coordinate = "y"
sample_col_name = f"{sample_keypoint}_{sample_coordinate}"

if sample_col_name in data.columns:
    sample_time_series = data[sample_col_name].dropna().values
    sample_troughs = detect_peaks_and_troughs(sample_time_series)

    # Ensure there are enough troughs to create segments
    if len(sample_troughs) > 1:
        for i in range(len(sample_troughs) - 1):
            start = sample_troughs[i]
            end = sample_troughs[i + 1]

            # Extract segment data for all keypoints
            segment = data.iloc[start:end + 1]

            # Save the segment to a new CSV file
            segment_file_name = f"rep_segment_{i + 1}.csv"
            segment.to_csv(os.path.join(output_segments_dir, segment_file_name), index=False)

            print(f"Segment {i + 1} saved: {segment_file_name}")

        # Plot segments for RIGHT_ELBOW_y as a sample
        sample_segment_plots = []
        for i in range(len(sample_troughs) - 1):
            start = sample_troughs[i]
            end = sample_troughs[i + 1]
            sample_segment_plots.append((data['frame_no'].iloc[start:end + 1].values, sample_time_series[start:end + 1]))

        plt.figure(figsize=(10, 6))
        for idx, (x, y) in enumerate(sample_segment_plots):
            plt.plot(x, y, label=f"Segment {idx + 1}")

        plt.title(f"Individual Reps (Segments): {sample_keypoint} ({sample_coordinate})")
        plt.xlabel('Frame Number')
        plt.ylabel(f'{sample_coordinate.upper()} Value')
        plt.legend()
        plt.grid()

        # Save the combined plot
        plot_file_name = f"segments_{sample_keypoint}_sample.png"
        plt.savefig(os.path.join(output_dir, plot_file_name))
        plt.close()

        print(f"Combined plot for {sample_keypoint} ({sample_coordinate}) saved: {plot_file_name}")
    else:
        print("Not enough troughs detected to extract segments.")
else:
    print(f"Sample column '{sample_col_name}' not found in the dataset.")
