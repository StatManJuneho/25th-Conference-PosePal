import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

file_name = "wrong"
# Input and output paths
file_path = f"test/{file_name}/{file_name}_pe_timeseries.csv"  # Path to your time-series CSV file
output_dir = f"test/{file_name}"
smoothed_csv = os.path.join(output_dir, f"{file_name}_smoothed_timeseries.csv")
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file
data = pd.read_csv(file_path)

# Filter columns for keypoints and coordinates
coordinate_columns = [col for col in data.columns if "_" in col and col != 'frame_no']

# Smooth and save data to a new CSV
smoothed_data = data.copy()
for col in coordinate_columns:
    time_series = data[col].dropna().values
    if len(time_series) > 10:  # Ensure there's enough data for smoothing
        smoothed_values = savgol_filter(time_series, window_length=31, polyorder=3)
        smoothed_data[col] = smoothed_values

smoothed_data.to_csv(smoothed_csv, index=False)
print(f"Smoothed data saved to: {smoothed_csv}")




### 여기 밑으로는 사실 없어도 되는 코드, peak detection은 다음 5번 파일에서 함
# Enhanced Trough Detection Function
def detect_peaks_and_troughs(column, time_axis, smoothed_series):
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
    refined_peaks = []
    refined_troughs = []

    # Check for valid peaks
    for peak in raw_peaks:
        if 0 < peak < len(smoothed_series) - 1:  # Ensure peak is not at the edge
            if smoothed_series[peak] > smoothed_series[peak - 1] and smoothed_series[peak] > smoothed_series[peak + 1]:
                refined_peaks.append(peak)

    # Check for valid troughs
    for trough in all_troughs:
        if 0 <= trough < len(smoothed_series) - 1:  # Include the first trough explicitly
            if (trough == 0 or smoothed_series[trough] < smoothed_series[trough - 1]) and \
               (trough == len(smoothed_series) - 1 or smoothed_series[trough] < smoothed_series[trough + 1]):
                refined_troughs.append(trough)

    # Return results
    return refined_peaks, refined_troughs

# Choose a column to validate
keypoint = "LEFT_ELBOW"
coordinate = "y"
col_name = f"{keypoint}_{coordinate}"  # Example: LEFT_ELBOW_y

if col_name in smoothed_data.columns:
    smoothed_time_series = smoothed_data[col_name].dropna().values
    time_axis = smoothed_data['frame_no']

    # Detect peaks and troughs
    peaks, troughs = detect_peaks_and_troughs(col_name, time_axis, smoothed_time_series)

    # Print results
    print("\nRefined Peak and Trough Detection Results:")
    print(f"Peaks (Indices): {peaks}")
    print(f"Troughs (Indices): {troughs}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, smoothed_time_series, label='Smoothed Time Series', color='blue')
    plt.plot(time_axis[peaks], smoothed_time_series[peaks], 'rx', label='Refined Peaks')
    plt.plot(time_axis[troughs], smoothed_time_series[troughs], 'bo', label='Refined Troughs')
    plt.title(f"Refined Peak and Trough Detection: {keypoint} ({coordinate})")
    plt.xlabel('Frame Number')
    plt.ylabel(f'{coordinate.upper()} Coordinate Value')
    plt.legend()
    plt.grid()

    # Save the plot
    plot_file_name = f"peak_detection_{keypoint}_enhanced.png"
    plt.savefig(os.path.join(output_dir, plot_file_name))
    plt.close()

    print(f"\nEnhanced validation plot saved to: {os.path.join(output_dir, plot_file_name)}")
else:
    print(f"Column '{col_name}' not found in the dataset.")
