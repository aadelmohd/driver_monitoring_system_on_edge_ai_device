import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

model = tf.keras.models.load_model("models/safe_unsafe_model.keras")

image_path = r"C:\Users\kitka\OneDrive\Desktop\archive\reduced_dataset\Train\unsafe\c0_unsafe_driving\img_3013.jpg"

img = load_img(image_path, target_size=(64, 64))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array, verbose=0)[0][0]

if prediction < 0.5:
    print(f"Prediction: safe ({1 - prediction:.4f})")
else:
    print(f"Prediction: unsafe ({prediction:.4f})")