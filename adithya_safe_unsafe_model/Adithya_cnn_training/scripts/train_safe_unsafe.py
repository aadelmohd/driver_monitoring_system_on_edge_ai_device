import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# path to dataset folder 
dataset_path = r"C:\Users\kitka\OneDrive\Desktop\archive\reduced_dataset\Train"

# images are set to fixed size
img_height = 64
img_width = 64
batch_size = 32
seed = 42

# list to store files for each class
safe_files = []
unsafe_files = []

# we collect all image paths from safe folder and its subfolders
for root, _, files in os.walk(os.path.join(dataset_path, "safe")):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            safe_files.append(os.path.join(root, f))
# similarly for unsafe folder
for root, _, files in os.walk(os.path.join(dataset_path, "unsafe")):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            unsafe_files.append(os.path.join(root, f))

all_files = safe_files + unsafe_files
all_labels = [0] * len(safe_files) + [1] * len(unsafe_files)

print(f"Safe images: {len(safe_files)}")
print(f"Unsafe images: {len(unsafe_files)}")
print(f"Total images: {len(all_files)}")

# we then split the dataset into training and validation sets (stratified split is used so both the classes stay balanced )
train_files, val_files, train_labels, val_labels = train_test_split(
    all_files,
    all_labels,
    test_size=0.2,
    random_state=seed,
    stratify=all_labels
)

print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")
print(f"Validation safe: {sum(1 for x in val_labels if x == 0)}")
print(f"Validation unsafe: {sum(1 for x in val_labels if x == 1)}")

# we use this function to load and preprocess a single image (it resizes the image and normalizes pixel values to the range [0,1])
def preprocess_image(path, label):
    img = load_img(path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = img / 255.0
    return img, np.array(label, dtype=np.float32)

# we load all the training images
x_train = []
y_train = []
for path, label in zip(train_files, train_labels):
    img, lbl = preprocess_image(path, label)
    x_train.append(img)
    y_train.append(lbl)

# we then load all validation images into memory 
x_val = []
y_val = []
for path, label in zip(val_files, val_labels):
    img, lbl = preprocess_image(path, label)
    x_val.append(img)
    y_val.append(lbl)

# this converts python lists into NumPy arrays for tensorflow training
x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
x_val = np.array(x_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)

# we use this to improve generalization
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# we use simple CNN model for binary image classification
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    data_augmentation,
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# then compile the model with Adam Optimizer and binary cross entropy loss
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# we train the model using the train set and then evaluate it on validation set
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=batch_size
)

os.makedirs("models", exist_ok=True)
model.save("models/safe_unsafe_model.keras")
print("Model saved as models/safe_unsafe_model.keras")

loaded_model = tf.keras.models.load_model("models/safe_unsafe_model.keras")
val_loss, val_acc = loaded_model.evaluate(x_val, y_val, verbose=1)
print(f"\nReloaded model validation accuracy: {val_acc:.4f}")

preds = loaded_model.predict(x_val, verbose=0)
pred_labels = (preds > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_val, pred_labels, target_names=["safe", "unsafe"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, pred_labels))

# we then save training curves for report use
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.savefig("results/training_curves.png")
plt.show()