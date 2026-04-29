import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Base folder: Adithya_cnn_training
base_dir = os.path.dirname(os.path.dirname(__file__))

# Model path
model_path = os.path.join(base_dir, "models", "safe_unsafe_model.keras")

# Test folder path
test_path = os.path.join(base_dir, "Test")

# Load trained model
model = tf.keras.models.load_model(model_path)

img_height = 64
img_width = 64

image_paths = []
true_labels = []

# Collect safe images
safe_folder = os.path.join(test_path, "safe")
for root, _, files in os.walk(safe_folder):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, f))
            true_labels.append(0)

# Collect unsafe images
unsafe_folder = os.path.join(test_path, "unsafe")
for root, _, files in os.walk(unsafe_folder):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, f))
            true_labels.append(1)

# Load test images
x_test = []
for path in image_paths:
    img = load_img(path, target_size=(img_height, img_width))
    img = img_to_array(img) / 255.0
    x_test.append(img)

x_test = np.array(x_test, dtype=np.float32)
true_labels = np.array(true_labels)

# Predict
pred_probs = model.predict(x_test, verbose=0).flatten()
pred_labels = (pred_probs >= 0.5).astype(int)

# Results
print("Test Accuracy:", accuracy_score(true_labels, pred_labels))
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=["safe", "unsafe"]))
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))