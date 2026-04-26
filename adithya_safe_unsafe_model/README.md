# Driver Monitoring on OpenMV

## Project Summary
This project implements a lightweight **driver monitoring system** for binary classification of driver behavior into **safe** and **unsafe** categories using an **edge AI workflow**.

The main objective was to build a complete end-to-end pipeline that:
- trains a model on laptop,
- evaluates its classification performance,
- converts the trained model into **TensorFlow Lite (TFLite)** format,
- and runs live inference on an **OpenMV camera**.

The implementation documented in this folder focuses on the **TensorFlow/Keras → TFLite → OpenMV** workflow.

---

## Team Members and Work Division

### Adithya
I was Responsible for the **CNN training and evaluation** stage:
- dataset handling for safe vs unsafe classification
- CNN model training using TensorFlow/Keras
- validation accuracy and confusion matrix analysis
- sample-image prediction testing
- saving the trained `.keras` model
- documenting model performance

### Vedic
My friend was Responsible for the **TFLite conversion and OpenMV deployment** stage:
- converting the trained CNN model to **TFLite**
- testing the `.tflite` model on laptop
- preparing deployment files for OpenMV
- running live inference on the OpenMV camera
- documenting deployment and edge testing

---

## Objectives
The work in this folder was designed to achieve the following:

1. Build a binary classifier for **safe vs unsafe driver behavior**
2. Train and evaluate a CNN model
3. Convert the trained CNN model to **TensorFlow Lite (TFLite)** format
4. Validate the converted TFLite model
5. Deploy the TFLite model to OpenMV
6. Run real-time safe/unsafe classification on the camera

---

## Dataset
A reduced driver behavior dataset was used for this work.

### Class Distribution
- **Safe images:** 1020  
- **Unsafe images:** 1200  
- **Total images:** 2220  

A **stratified 80/20 train-validation split** was used to preserve class balance during training and evaluation.

---

## CNN Model Development
The core classification model used in this project is a **Convolutional Neural Network (CNN)** trained using **TensorFlow/Keras**.

### Training Setup
- Task: binary classification
- Classes:
  - `safe`
  - `unsafe`
- Image size: `64 × 64`
- Batch size: `32`

### Data Augmentation
To improve generalization, the following augmentation methods were applied:
- horizontal flip
- slight rotation
- slight zoom

---

## Final CNN Performance

### Validation Accuracy
- **90.32%**

### Classification Report
- **Safe**
  - Precision: `0.90`
  - Recall: `0.88`
  - F1-score: `0.89`

- **Unsafe**
  - Precision: `0.90`
  - Recall: `0.92`
  - F1-score: `0.91`

### Confusion Matrix
```text
[[180, 24],
 [ 19, 221]]