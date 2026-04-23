# Run this once to install the required package if not already installed:
# pip install ultralytics

import os
import shutil


BASE_PATH = r"C:\Users\vedic\OneDrive\Desktop\deep learning\dataset"

src_images = os.path.join(BASE_PATH, "images")
src_labels = os.path.join(BASE_PATH, "labels")
dst_root = os.path.join(BASE_PATH, "dataset_cls")

splits = ["train", "val"]

print(" Using dataset at:", BASE_PATH)

if not os.path.exists(src_images):
    raise FileNotFoundError(f"Images folder not found: {src_images}")

if not os.path.exists(src_labels):
    raise FileNotFoundError(f"Labels folder not found: {src_labels}")

print(" Paths verified\n")

for split in splits:
    img_dir = os.path.join(src_images, split)
    lbl_dir = os.path.join(src_labels, split)

    if not os.path.exists(img_dir):
        print(f" Missing: {img_dir}")
        continue

    if not os.path.exists(lbl_dir):
        print(f" Missing: {lbl_dir}")
        continue

    print(f" Processing {split}...")

    for file in os.listdir(img_dir):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(img_dir, file)
        label_path = os.path.join(lbl_dir, file.rsplit(".", 1)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        # Read class ID
        with open(label_path, "r") as f:
            line = f.readline().strip()
            if not line:
                continue
            class_id = line.split()[0]

        class_folder = os.path.join(dst_root, split, f"class_{class_id}")
        os.makedirs(class_folder, exist_ok=True)

        # Copy image
        shutil.copy(img_path, os.path.join(class_folder, file))

print("\n Dataset converted successfully → dataset_cls")
