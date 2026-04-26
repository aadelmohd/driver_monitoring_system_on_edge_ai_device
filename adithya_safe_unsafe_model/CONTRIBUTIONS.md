# Contributions

## Adithya
Responsible for the CNN training and evaluation stage.

### Work completed
- Prepared and used the reduced safe vs unsafe dataset
- Trained the CNN model using TensorFlow/Keras
- Evaluated model performance using validation accuracy, classification report, and confusion matrix
- Created scripts for:
  - training
  - single-image prediction
  - folder-level prediction checks
- Saved the trained `.keras` model
- Documented training results

### Files
- `Adithya_cnn_training/scripts/train_safe_unsafe.py`
- `Adithya_cnn_training/scripts/predict_image.py`
- `Adithya_cnn_training/scripts/check_folder_predictions.py`
- `Adithya_cnn_training/models/safe_unsafe_model.keras`
- `Adithya_cnn_training/results/training_curves.png`
- `Adithya_cnn_training/results/accuracy_ss.png`
- `Adithya_cnn_training/notes/model_results.md`

---

## Vedic
Responsible for the TFLite conversion and OpenMV deployment stage.

### Work completed
- Converted the trained CNN model to `.tflite`
- Tested TFLite inference on safe and unsafe sample images
- Prepared deployment files for OpenMV
- Ran live safe/unsafe inference on the OpenMV camera
- Documented deployment results

### Files
- `Vedic_tflite_openmv/scripts/convert_to_tflite.py`
- `Vedic_tflite_openmv/scripts/test_tflite.py`
- `Vedic_tflite_openmv/models/safe_unsafe_model.tflite`
- `Vedic_tflite_openmv/models/labels.txt`
- `Vedic_tflite_openmv/openmv/main.py`
- `Vedic_tflite_openmv/results/tflite_safe_test.png`
- `Vedic_tflite_openmv/results/tflite_unsafe_test.png`
- `Vedic_tflite_openmv/results/openmv_safe_result.png`
- `Vedic_tflite_openmv/results/openmv_unsafe_result.png`
- `Vedic_tflite_openmv/notes/deployment_notes.md`