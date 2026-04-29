import os
import shutil
import random

source_dir = "dataset"
output_dir = "dataset_split"

classes = ["safe", "unsafe"]

split_ratio = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

# Create folders
for split in split_ratio:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Split files
for cls in classes:
    files = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(files)

    total = len(files)
    train_end = int(split_ratio["train"] * total)
    val_end = int((split_ratio["train"] + split_ratio["val"]) * total)

    for i, file in enumerate(files):
        src = os.path.join(source_dir, cls, file)

        if i < train_end:
            dst = os.path.join(output_dir, "train", cls, file)
        elif i < val_end:
            dst = os.path.join(output_dir, "val", cls, file)
        else:
            dst = os.path.join(output_dir, "test", cls, file)

        shutil.copy(src, dst)

print("Dataset split complete!")