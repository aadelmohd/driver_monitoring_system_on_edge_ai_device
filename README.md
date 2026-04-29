# Driver Monitoring System on Edge AI Device

## Dataset
Kaggle dataset: https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data

## Reduced dataset
https://kaggle.com/datasets/4f05318ebd2cbd6c94d60c2d2484304e0a076a27fa223555e7863ed65b48c587


Note: Use this dataset for the project, but currently use only classes C0 to C4; the rest can be used later.
Make sure to re-label the classes into safe driving and unsafe driving for the baseline model.


## Project Summary
The project implements a lightweight driver monitoring system for binary classification of driver behaviour into safe and unsafe categories based on cell phone usage in an Edge AI module OpenMV Cam RT1062.

The goals of this project are:
- Train various models
- Compare the model performance trained in Edge Impulse vs Local Device
- Evaluate the models
- Convert to TFLite int8 format (Quantisation)
- Deploy on OpenMV module
- Test the best model on a car
- As a whole build a driver monitoring system 
			
The main objective was to build a system that encompasses:
•	Best trained model either on local device or Edge Impulse
•	Runs live inference on an OpenMV camera module.
The implementation documented in this folder focuses on the TensorFlow/Keras → TFLite → OpenMV workflow.

---

## Results

| S.No. | Model | Ecosystem | Test Accuracy (%) |
| :--- | :--- | :--- | :--- |
| 1. | EfficientNet | Edge Impulse | 86.36 |
| 2. | MobileNet v1 | Edge Impulse | 31.82 |
| 3. | CNN Multi-class | Edge Impulse | 85.88 |
| 4. | CNN (Data Augmented) | Local Device - CPU | 91.67 |
| 5. | FOMO Multi-class | Edge Impulse | 91.00 |
| 6. | FOMO Multi-class (Data Augmented) | Edge Impulse | 90.83 |
| 7. | CNN Binary | Edge Impulse | 99.12 |
| 8. | CNN Binary (Data Augmented) | Local Device - CPU | 86.19 |

---

### Best Model:
#### CNN Binary (Edge Impulse) 
Test Accuracy: 99.12



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
