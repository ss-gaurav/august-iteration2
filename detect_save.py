import cv2
import time
import os
import pandas as pd
from ultralytics import YOLO

# Set paths
video_folder = 'videos'  # Path to the folder with your videos
model_folder = 'models'  # Path to the folder with your YOLOv8 models
output_folder = 'outputs'  # Path to save output videos
csv_log_file = os.path.join(output_folder, 'log.csv')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all videos and models
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
model_files = [f for f in os.listdir(model_folder) if f.endswith(('.pt', '.engine'))]

# Initialize log dataframe
log_columns = ['Model', 'Video', 'Total_Frames', 'Total_Time_Seconds', 'Average_FPS']
log_data = []

print(video_files)
print(model_files)



# Target resolution for 720p
target_resolution = (1920, 2160)

start_time = time.time()

# Loop through each model
for model_file in model_files:
    model_path = os.path.join(model_folder, model_file)
    model = YOLO(model_path)

    # Loop through each video
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Prepare output video file
        output_video_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_{os.path.splitext(model_file)[0]}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, target_resolution)

        start_time = time.time()
        frame_count = 0

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 720p
            resized_frame = cv2.resize(frame, target_resolution)

            # Run YOLOv8 inference on the resized frame
            results = model(resized_frame)

            # Draw results on the frame (optional)
            annotated_frame = results[0].plot()
            
            #resize annotated frame
            #resized_annotated_frame = cv2.resize(frame, target_resolution)

            # Write the frame to the output video
            #out.write(annotated_frame)
            frame_count += 1

        # Calculate total time and average FPS
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time

        # Release resources
        cap.release()
        out.release()

        # Log details
        log_data.append([model_file, video_file, frame_count, total_time, avg_fps])

end_time = time.time()

print("TOTAL TIME", end_time-start_time)
# Save logs to CSV
df_log = pd.DataFrame(log_data, columns=log_columns)
df_log.to_csv(csv_log_file, index=False)

print(f"Processing complete. Logs saved to {csv_log_file}")
