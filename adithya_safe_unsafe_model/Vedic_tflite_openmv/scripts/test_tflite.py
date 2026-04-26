import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Path to one test image
image_path = r"C:\Users\kitka\OneDrive\Desktop\archive\reduced_dataset\Train\safe\c1_safe_driving\img_2545.jpg"
print("Testing image:", image_path)
# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="Vedic_tflite_openmv/models/safe_unsafe_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read model input size
input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# Load and preprocess image
img = load_img(image_path, target_size=(height, width))
img = img_to_array(img)
img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], img)

# Run inference
interpreter.invoke()

# Read output
output = interpreter.get_tensor(output_details[0]['index'])[0][0]

if output < 0.5:
    print(f"Prediction: safe ({1 - output:.4f})")
else:
    print(f"Prediction: unsafe ({output:.4f})")