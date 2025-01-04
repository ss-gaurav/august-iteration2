import cv2
import threading
import os
import serial
import torch
import queue
from queue import Queue
from ultralytics import YOLO
from datetime import datetime
import time

# Configuration parameters
show_stream = False      # Show camera stream
save_video = False       # Save video
weights = "bin/drone_nano.pt"  # Weights file
detection_mode = "locking"  # Initial detection mode
detection_conf_threshold = 0.40  # Confidence threshold for switching to chasing mode
lock_threshold = 0.20  # Confidence threshold for switching back to locking mode
detection_count_threshold = 5  # Number of detections for mode switch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Load YOLO model
model = YOLO(weights)

# Setup serial communication (Using the Jetson Nano's UART port)
try:
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    ser.readall()
    print("Serial port opened successfully")
except Exception as e:
    print(f"Failed to open serial port: {e}")

# GStreamer pipeline function
def gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=60, flip_method=0):
    return (
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        "format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
    "nvvidconv ! video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)
# Function to capture the latest frame
def capture_latest_frame(cap, frame_holder, stop_event):
    while cap.isOpened() and not stop_event.is_set():
        ret_val, img = cap.read()
        if not ret_val:
            print("Failed to capture frame")
            continue
        frame_holder['frame'] = img

# Function to generate unique filename
def generate_unique_filename(base_name, extension=".avi"):
    i = 1
    while os.path.exists(f"video/{base_name}{i}{extension}"):
        i += 1
    return f"video/{base_name}{i}{extension}"

# Function to save frames in a separate thread
def save_frames(out, frame_queue, stop_event):
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break
            out.write(frame)
            frame_queue.task_done()
        except queue.Empty:
            continue

# Function to continuously read from the serial port
def read_serial_data(serial_port, received_data_queue, stop_event):
    while not stop_event.is_set():
        try:
            data = serial_port.readline().decode('utf-8').rstrip()
            if data:
                received_data_queue.put(data)
        except serial.SerialTimeoutException:
            continue

if __name__ == "__main__":
    save_quality = (640, 480)  # Ensure this matches the frame size you're capturing
    save_fps = 20
    save_name = "dumdum"  # Base name for the video and log files

    # GStreamer pipeline and video capture setup
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Failed to open camera")
    else:
        frame_holder = {'frame': None}
        frame_queue = Queue(maxsize=5)
        received_data_queue = Queue()
        stop_event = threading.Event()

        # Generate video and log filenames
        video_filename = generate_unique_filename(save_name)
        log_filename = video_filename.replace(".avi", ".txt")

        # Open the log file for writing
        with open(log_filename, 'w') as log_file:

            # Write initial log details
            log_file.write(f"Log for {video_filename}\n")
            log_file.write(f"Start Time: {datetime.now()}\n")
            log_file.write(f"Camera Settings: {save_quality[0]}x{save_quality[1]}, {save_fps} FPS\n")
            log_file.write(f"Weights used: {weights}\n")
            log_file.write("-" * 50 + "\n")

            # Start threads for capturing frames and reading serial data
            capture_thread = threading.Thread(target=capture_latest_frame, args=(cap, frame_holder, stop_event))
            capture_thread.start()

            serial_thread = threading.Thread(target=read_serial_data, args=(ser, received_data_queue, stop_event))
            serial_thread.start()

            out = None
            if save_video:
                # Open the VideoWriter with the correct settings for color
                out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), save_fps, save_quality, isColor=True)
                if not out.isOpened():
                    print("Failed to open VideoWriter")

            detection_count = 0  # Count of valid detections
            recent_confidences = []  # Store recent confidence scores



            # Inside the main loop
            try:
                while True:
                    loop_start = time.time()  # Record the loop start time

                    serial_data = "0,0,0\n"  # Default data to send if no valid detection occurs

                    if frame_holder['frame'] is not None:
                        img = frame_holder['frame']
                        results = model(img, stream=True)

                        best_box = None  # Reset best box for each loop
                        for r in results:
                            for box in r.boxes:
                                confidence = box.conf.item()
                                recent_confidences.append(confidence)

                                log_file.write(f"Detected object with confidence: {confidence}\n")

                                if confidence > detection_conf_threshold:
                                    detection_count += 1
                                else:
                                    detection_count = max(0, detection_count - 1)

                                if detection_count >= detection_count_threshold:
                                    detection_mode = "chasing"
                                elif (
                                    len(recent_confidences) >= detection_count_threshold
                                    and all(c < lock_threshold for c in recent_confidences[-detection_count_threshold:])
                                ):
                                    detection_mode = "locking"

                                log_file.write(f"Mode: {detection_mode.upper()}\n")

                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                                cv2.putText(img, f'Object: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                                log_file.write(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}\n")

                                if detection_mode == "chasing" and (best_box is None or confidence > best_box.conf.item()):
                                    best_box = box

                        if best_box and detection_mode == "chasing":
                            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            radius = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 2)
                            cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 2)

                            area = (x2 - x1) * (y2 - y1)
                            serial_data = f"{center_x},{center_y},{area},\n"  # Update valid data to send
                            log_file.write(f"Circle drawn at: center=({center_x},{center_y}), radius={radius}, area={area}\n")

                        cv2.putText(img, f"Mode: {detection_mode.upper()}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        if not received_data_queue.empty():
                            received_data = received_data_queue.get()
                            print(f"Received: {received_data}")
                            cv2.putText(img, received_data, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
                            log_file.write(f"Received serial data: {received_data}\n")

                        if save_video and out:
                            resized_frame = cv2.resize(img, save_quality)
                            try:
                                frame_queue.put_nowait(resized_frame)
                            except queue.Full:
                                print("Frame skipped to maintain processing speed.")
                            out.write(resized_frame)

                        if show_stream:
                            cv2.imshow("CSI Camera", resized_frame)

                    # Ensure each loop takes at least 50 ms
                    loop_duration = time.time() - loop_start
                    if loop_duration < 0.05:
                        time.sleep(0.05 - loop_duration)

                    # Send data at the end of the loop
                    ser.write(serial_data.encode())
                    print(f"Sent: {serial_data.strip()}")

                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            finally:
                stop_event.set()
                capture_thread.join()
                serial_thread.join()
                if save_video and out:
                    save_thread.join()
                cap.release()
                if out:
                    out.release()
                cv2.destroyAllWindows()
                ser.close()
                log_file.write(f"End Time: {datetime.now()}\n")
                log_file.write("-" * 50 + "\n")
