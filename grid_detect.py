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

#Detection image size 

image_size = 640

# Target resolution for 720p


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
        target_resolution = (width, height)
        N = width // image_size
        M = height // image_size

        print (f'video size = {width}, {height}, N = {N}, M = {M}')

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
            
            num_of_detections = 0

            detected_box = []
            
            for n in range (N):  # N is Hidth and M is Height
                for m in range (M):
                    
                    print (f'height - {height}, width - {width}, N - {N}, M - {M}, n - {n}, m - {m}')
                    roi = frame[m*image_size:(m+1)*image_size, n*image_size:(n+1)*image_size]   #frame[y1:y2, x1:x2]
                    offset = (n*image_size , m*image_size)

                    # Run YOLOv8 inference on the resized frame
                    results = model(roi , imgsz = image_size)

                    for result in results:
                        # Each `result.boxes` contains bounding box information
                        for box in result.boxes:
                            # Extract coordinates, confidence, and class label
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box in x1, y1, x2, y2 format
                            
                            
                            confidence = box.conf.cpu().item()  # Confidence score
                            class_id = box.cls.cpu().item()  # Class ID

                            num_of_detections +=1

                            X1, X2, Y1, Y2 = int (x1 + offset[0]), int (x2 + offset[0]), int (y1 + offset[1]), int (y2 + offset[1])
                            detected_box.append ((X1+X2)//2)
                            print (X1,X2,Y1,Y2)

                            frame = cv2.rectangle(frame , (X1,Y1), (X2,Y2), (255,0,0), 2)

                            print (f'n is {n} m is {m}')

            if len(detected_box) == 2:
                disparity = detected_box[0] - detected_box[1]
                cv2.putText(frame,  f'disparity = {disparity}',(50, 100),  cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 255),    2, cv2.LINE_4) 
                cv2.line(frame, (int(detected_box[0]), 0), (int(detected_box[0]), height), (0,255,0), 2)
                cv2.line(frame, (int(detected_box[1]), 0), (int(detected_box[1]), height), (0,255,0), 2)
                cv2.rectangle(frame, ((int(detected_box[0]), height // 2 )), (int(detected_box[1]), height//2 + 20) , (255, 0,0), -1)

            cv2.putText(frame,  f'number of detection : {num_of_detections}',(50, 50),  cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 255),    2, cv2.LINE_4) 

            # Write the frame to the output video       
            out.write(frame)
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
