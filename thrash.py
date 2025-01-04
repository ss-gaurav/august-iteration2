from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")  # Load a model
model.export(format="engine", int8=True)