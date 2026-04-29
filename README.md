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

### Tirthraj
Responsible for the **Data preparation and Edge Impulse CNN Model training**:
- dataset portioned and uploaded to kaggle as "Reduced Dataset"
- trained a CNN model on Edge Impulse
- overlooked tuning using Edge Impulse
- exported the trained model into tflite
- ran the testing with a car while running the OpenMV Script

### Aadel
Responsible for the **Data Collection, EfficientNet training and Local CNN training**:
- dataset was collected from Kaggle\
- the Reduced Dataset was used to preprocess into clear directories
- a Python script was made to split the dataset into train, val, and test (70,15,15) sub-folders.
- build the Edge Impulse CNN model from scratch in VSCode
- performed data augmentation
- data rescaling for normalisation
- model trained with the same lerning rate and epochs as edge impulse
- loss and accuracy was calculated based on test set
- confusion matrix generated with test set
- classification report generated

### Eshwar:
Responsible for ***Quantisation and Accelerometer testing***:
- the CNN model was quantized into TFlite int8 format
- tested the accelerometer
- the detection of speed could not be detected using this module

### Adithya
Responsible for the **CNN training and evaluation** stage:
- dataset handling for safe vs unsafe classification
- CNN model training using TensorFlow/Keras
- validation accuracy and confusion matrix analysis
- sample-image prediction testing
- saving the trained `.keras` model
- documenting model performance
- Performed data augmentation using transfer learning in Edge Impulse.
- Improved the model’s ability to generalise from a limited dataset.
- Used the FOMO model, which is lightweight and suitable for edge devices.
- Uploaded the trained model to the team’s GitHub repository.

### Vedic
Responsible for the **TFLite conversion and OpenMV deployment** stage:
- converting the trained CNN model to **TFLite**
- testing the `.tflite` model on laptop
- preparing deployment files for OpenMV
- running live inference on the OpenMV camera
- documenting deployment and edge testing
- focused on validation and real-world output testing.
- tested the trained model using driver images.
- used images of drivers using a mobile phone while driving to check unsafe behaviour detection.
- worked on improving the model’s accuracy based on the testing results.
- implemented the red and green LED indicator system on the edge device.
- used the green LED to indicate a safe driver and the red LED to indicate unsafe behaviour.
