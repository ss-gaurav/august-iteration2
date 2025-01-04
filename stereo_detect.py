import cv2
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Set paths
video_folder = 'videos'  # Path to the folder with your videos
model_folder = 'models'  # Path to the folder with your YOLOv8 models
output_folder = 'outputs'  # Path to save output videos
csv_log_file = os.path.join(output_folder, 'log.csv')
distance_file = '/home/ss/drone-detection/detection-testing/output_2024-10-16_18-08-07.txt'  # Your provided text file with distances

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all videos and models
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
model_files = [f for f in os.listdir(model_folder) if f.endswith(('.pt', '.engine'))]

# Read distances from the text file
distances = []
with open(distance_file, 'r') as f:
    for line in f:
        if "Distance" in line:
            # Extract the distance value from the line
            distance_value = float(line.split("Distance:")[1].split("m")[0].strip())
            distances.append(distance_value)

# Initialize log dataframe
log_columns = ['Model', 'Video', 'Total_Frames', 'Total_Time_Seconds', 'Average_FPS']
log_data = []
x_diff_values = []  # To store (640 - x_diff) values
filtered_distances = []  # To store corresponding distances for valid x_diff values

# Prepare to save output text
output_text_file_path = os.path.join(output_folder, 'output.txt')
with open(output_text_file_path, 'w') as text_file:

    # Loop through each model
    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        model = YOLO(model_path)

        # Loop through each video
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Prepare output video file
            output_video_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_{os.path.splitext(model_file)[0]}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (1280, 480))

            start_time = time.time()
            frame_count = 0

            # Process video frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to target resolution
                cropped_image = frame[0:480, 0:1280]

                resized_frame = cv2.resize(frame, (960, 1080))
                height, width, _ = resized_frame.shape

                # Run YOLOv8 inference on the resized frame
                results = model(cropped_image)

                # Get bounding boxes from results
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Move tensor to CPU and convert to NumPy array

                # Handle frames with at least 2 detections
                if len(boxes) >= 2:
                    # Calculate centers of the bounding boxes
                    centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2  # Average of x1,y1 and x2,y2

                    # Calculate the difference in the x-coordinates of the centers
                    x_diff = abs(centers[0, 0] - centers[1, 0])  # Difference between centers
                    adjusted_x_diff = 640 - x_diff  # Calculate (640 - x_diff)
                    x_diff_values.append(adjusted_x_diff)  # Store for later plotting

                    # Write x_diff to the text file
                    text_file.write(f"X coordinate difference: {x_diff:.2f}\n")

                    # Draw the difference on the frame
                    cv2.putText(cropped_image, f"X Diff: {adjusted_x_diff:.2f}", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    # Store the corresponding distance if it exists
                    if frame_count < len(distances):
                        filtered_distances.append(distances[frame_count])

                frame_count += 1
                # Draw results on the frame (optional)
                annotated_frame = results[0].plot()

                # Write the frame to the output video
                out.write(annotated_frame)

            # Calculate total time and average FPS
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time

            # Release resources
            cap.release()
            out.release()

            # Log details
            log_data.append([model_file, video_file, frame_count, total_time, avg_fps])

# Save logs to CSV
df_log = pd.DataFrame(log_data, columns=log_columns)
df_log.to_csv(csv_log_file, index=False)

# Align lengths of filtered_distances and x_diff_values
min_length = min(len(filtered_distances), len(x_diff_values))
filtered_distances = filtered_distances[:min_length]
x_diff_values = x_diff_values[:min_length]

# Plot the graph between filtered_distances and (640 - x_diff)
plt.figure(figsize=(12, 6))
plt.plot(filtered_distances, x_diff_values, marker='o', linestyle='-', color='b')
plt.title('Graph between Distance and (640 - x_diff)')
plt.xlabel('Distance (m)')
plt.ylabel('(640 - x_diff)')
plt.grid()
plt.show()

print(f"Processing complete. Logs saved to {csv_log_file} and text output saved to {output_text_file_path}")

