import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
import random

# we load the trained model
model = tf.keras.models.load_model("models/safe_unsafe_model.keras")

safe_dir = r"C:\Users\kitka\OneDrive\Desktop\archive\reduced_dataset\Train\safe"
unsafe_dir = r"C:\Users\kitka\OneDrive\Desktop\archive\reduced_dataset\Train\unsafe"

def get_all_images(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(root, f))
    return files

def predict_image(path):
    img = load_img(path, target_size=(64, 64))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    score = float(model.predict(img, verbose=0)[0][0])
    label = "unsafe" if score >= 0.5 else "safe"
    return label, score

# we check random sample of images from both the classes to check wether it is safe or unsafe
for folder_name, folder_path in [("SAFE", safe_dir), ("UNSAFE", unsafe_dir)]:
    files = get_all_images(folder_path)
    sample_files = random.sample(files, min(20, len(files)))
    print(f"\nTesting {folder_name}:")
    for path in sample_files:
        pred_label, score = predict_image(path)
        print(f"{os.path.basename(path)} -> {pred_label} (raw={score:.4f})")