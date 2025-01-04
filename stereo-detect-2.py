import cv2
import os
import pandas as pd
import time
from ultralytics import YOLO

# Set paths
video_folder = 'videos'  # Path to the folder with your videos
model_folder = 'models'  # Path to the folder with your YOLOv8 models
output_folder = 'outputs'  # Path to save output videos
csv_log_file = os.path.join(output_folder, 'drone_detections.csv')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all videos and models
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
model_files = [f for f in os.listdir(model_folder) if f.endswith(('.pt', '.engine'))]

# Initialize dataframe for logging
log_columns = ['Model', 'Video', 'Frame', 'Top_Left_Center', 'Top_Left_Coords', 'Top_Right_Center', 'Top_Right_Coords', 'Disparity']
detection_log = []

# Loop through each model
for model_file in model_files:
    model_path = os.path.join(model_folder, model_file)
    model = YOLO(model_path)

    # Loop through each video
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        output_video_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_{os.path.splitext(model_file)[0]}_visualized.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Split frame into sections
            frame_height, frame_width = frame.shape[:2]
            top_left = frame[0:frame_height//2, 0:frame_width//2]
            top_right = frame[0:frame_height//2, frame_width//2:frame_width]

            # Initialize centers and disparity
            top_left_center = (0, 0)
            top_right_center = (0, 0)
            disparity = None

            # Run YOLOv8 inference on the top left section
            results = model(top_left)
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes

            if len(boxes) > 0:
                # Assume the first detected box is the drone
                x1, y1, x2, y2 = boxes[0]
                top_left_coords = (int(x1), int(y1), int(x2), int(y2))
                top_left_center = ((x1 + x2) / 2, (y1 + y2) / 2)

                # Draw bounding box and center on the top left section
                cv2.rectangle(top_left, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(top_left, (int(top_left_center[0]), int(top_left_center[1])), 5, (0, 255, 0), -1)

                # Extract the detected drone as a template
                template = top_left[int(y1):int(y2), int(x1):int(x2)]
                template_height, template_width = template.shape[:2]

                # Define the region of interest (ROI) for template matching in the top right section
                roi_width = int(3 * (x2 - x1))
                roi_height = int(3 * (y2 - y1))
                roi_center_x = int(top_left_center[0])
                roi_center_y = int(top_left_center[1])

                # Calculate the ROI coordinates based on the center position
                roi_x1 = max(0, roi_center_x - roi_width // 2)
                roi_y1 = max(0, roi_center_y - roi_height // 2)
                roi_x2 = min(frame_width // 2, roi_x1 + roi_width)
                roi_y2 = min(frame_height // 2, roi_y1 + roi_height)

                # Visualize the ROI in the top right section with a black boundary
                cv2.rectangle(top_right, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 0), 2)

                # Limit the top right image to the defined ROI
                roi_top_right = top_right[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_height, roi_width = roi_top_right.shape[:2]

                # Check if the template fits within the ROI
                if template_height <= roi_height and template_width <= roi_width:
                    # Perform template matching in the ROI of the top right section
                    result = cv2.matchTemplate(roi_top_right, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    matched_x1, matched_y1 = max_loc
                    matched_x1 += roi_x1
                    matched_y1 += roi_y1
                    matched_x2 = matched_x1 + template_width
                    matched_y2 = matched_y1 + template_height
                    top_right_coords = (matched_x1, matched_y1, matched_x2, matched_y2)
                    top_right_center = ((matched_x1 + matched_x2) / 2, (matched_y1 + matched_y2) / 2)

                    # Draw bounding box and center on the top right section
                    cv2.rectangle(top_right, (matched_x1, matched_y1), (matched_x2, matched_y2), (0, 0, 255), 2)
                    cv2.circle(top_right, (int(top_right_center[0]), int(top_right_center[1])), 5, (0, 0, 255), -1)

                    # Calculate disparity if both centers are available
                    disparity = top_left_center[0] - top_right_center[0]

            # Display the disparity on the frame
            disparity_text = f"Disparity: {disparity if disparity is not None else 'N/A'}"
            cv2.putText(frame, disparity_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Log data
            detection_log.append([
                model_file, video_file, frame_count,
                top_left_center, top_left_coords if 'top_left_coords' in locals() else (0, 0, 0, 0),
                top_right_center, top_right_coords if 'top_right_coords' in locals() else (0, 0, 0, 0),
                disparity
            ])

            # Combine the modified sections back into the original frame
            frame[0:frame_height//2, 0:frame_width//2] = top_left
            frame[0:frame_height//2, frame_width//2:frame_width] = top_right

            # Write the visualized frame to the output video
            out.write(frame)

            frame_count += 1

        # Release resources
        cap.release()
        out.release()

# Save detection log to CSV
df_log = pd.DataFrame(detection_log, columns=log_columns)
df_log.to_csv(csv_log_file, index=False)

print(f"Processing complete. Detections saved to {csv_log_file} and visualized videos saved in {output_folder}.")
