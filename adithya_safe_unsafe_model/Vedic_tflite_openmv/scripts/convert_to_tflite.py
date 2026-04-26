import tensorflow as tf
import os

model = tf.keras.models.load_model("models/safe_unsafe_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

os.makedirs("models", exist_ok=True)
with open("models/safe_unsafe_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as models/safe_unsafe_model.tflite")