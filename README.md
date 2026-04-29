# Driver Monitoring System on Edge AI Device

## Dataset
Kaggle dataset: https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data

## Reduced dataset
https://kaggle.com/datasets/4f05318ebd2cbd6c94d60c2d2484304e0a076a27fa223555e7863ed65b48c587


Note: Use this dataset for the project, but currently use only classes C0 to C4; the rest can be used later.
Make sure to re-label the classes into safe driving and unsafe driving for the baseline model.



The accuracy os yolo model is 74%. it predicts the two classes you are using phone or not while driving


## Project Summary
This project implements a lightweight **driver monitoring system** for binary classification of driver behavior into **safe** and **unsafe** categories using an **edge AI workflow**.

The main objective was to build a complete end-to-end pipeline that:
- trains a model on local device/Edge Impulse, 
- evaluates its classification performance,
- converts the trained model into **TensorFlow Lite (TFLite)** format,
- and runs live inference on an **OpenMV camera module**.

The implementation documented in this folder focuses on the **TensorFlow/Keras → TFLite → OpenMV** workflow.

---

## Team Members and Work Division

### Adithya
Responsible for the **CNN training and evaluation** stage:
- dataset handling for safe vs unsafe classification
- CNN model training using TensorFlow/Keras
- validation accuracy and confusion matrix analysis
- sample-image prediction testing
- saving the trained `.keras` model
- documenting model performance

### Vedic
Responsible for the **TFLite conversion and OpenMV deployment** stage:
- converting the trained CNN model to **TFLite**
- testing the `.tflite` model on laptop
- preparing deployment files for OpenMV
- running live inference on the OpenMV camera
- documenting deployment and edge testing
