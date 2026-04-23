import os
from ultralytics import YOLO


PROJECT_PATH = "C:/Users/vedic/OneDrive/Desktop/deep learning"

# Automatically find best.pt
best_model_path = None

for root, dirs, files in os.walk(PROJECT_PATH):
    if "best.pt" in files:
        best_model_path = os.path.join(root, "best.pt")
        break

# Check if found
if best_model_path is None:
    raise FileNotFoundError("best.pt not found in project folder!")

print("Model found at:", best_model_path)

# Load model
model = YOLO(best_model_path)

# Run validation to get accuracy
metrics = model.val(data="dataset")

# Print accuracy
print(f"Top-1 Accuracy: {metrics.top1:.4f}")
print(f"Top-5 Accuracy: {metrics.top5:.4f}")
