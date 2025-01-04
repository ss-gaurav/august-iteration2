from ultralytics import YOLO
import os

model = YOLO('models/best.pt')
model.export(format='engine', int8=True)

old_path = "models/best.engine"
new_path = "models/best_int8.engine"

if os.path.exists(new_path):
    os.remove(new_path)
    print(f"Deleted existing file: {new_path}")

os.remove("models/best.cache")
os.rename(old_path, new_path)

print(f"File renamed to {new_path}")
