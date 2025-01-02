####
# 시각화를 위한 함수, 딱히 필요하지는 않음
###

import pandas as pd
import matplotlib.pyplot as plt
import os

# Reload the dataset
file_name = "wrong"
file_path = f"test/{file_name}/{file_name}_pe_timeseries.csv"  # Path to your time-series CSV file
data = pd.read_csv(file_path)

# Output directory for saving the plot
output_dir = f"test/{file_name}"
os.makedirs(output_dir, exist_ok=True)

# Extract frame numbers for x-axis
frame_numbers = data['frame_no']

# Define the most important key points
# key_points = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST']
key_points = ['LEFT_ELBOW', 'RIGHT_ELBOW']
coordinates = ['y']  # Coordinates to consider

# Plot only the important key points
plt.figure(figsize=(15, 10))

for key_point in key_points:
    for coord in coordinates:
        column_name = f"{key_point}_{coord}"
        if column_name not in data.columns:
            print(f"Skipping missing column: {column_name}")
            continue
        
        # Plot each coordinate with a label
        plt.plot(frame_numbers, data[column_name], label=f'{key_point} {coord.upper()}')

# Add title, labels, and legend
plt.title("Side Lateral Raise - Key Points Time Series", fontsize=16)
plt.xlabel("Frame Number", fontsize=14)
plt.ylabel("Coordinate Values", fontsize=14)
plt.legend(loc='upper right', fontsize=10, ncol=2)  # Adjust legend size and columns

# Save the plot to a file
output_file = os.path.join(output_dir, "side_lateral_raise_plot.png")
plt.savefig(output_file)
plt.close()

print(f"Plot focusing on key points saved as: {output_file}")
