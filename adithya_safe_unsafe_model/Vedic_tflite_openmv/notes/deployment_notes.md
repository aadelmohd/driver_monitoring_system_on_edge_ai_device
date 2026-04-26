# Deployment Notes

## Current status
- OpenMV camera connection was tested successfully
- `main.py` runs on the OpenMV board and confirms live camera feed
- Safe vs unsafe image classification model was trained on laptop
- Final TensorFlow/Keras model saved as `safe_unsafe_model.keras`

## Model choice
- Main model used: CNN-based binary classifier
- Task: classify driver images into:
  - safe
  - unsafe

## Why this model was selected
- Produced stable validation accuracy
- Easier to train and evaluate than failed Edge Impulse attempt
- Suitable as a first deployable baseline for OpenMV/TFLite workflow

## Additional testing
- Folder-level testing was added using:
  - `check_folder_predictions.py`
- Single-image inference was added using:
  - `predict_image.py`

## Next deployment step
- Convert `.keras` model to `.tflite`
- Test the `.tflite` model on laptop
- Then attempt OpenMV deployment if compatible

## TFLite conversion check
- Converted `safe_unsafe_model.keras` to `safe_unsafe_model.tflite`
- Tested TFLite model on laptop before edge deployment

### Example TFLite inference results
- Safe image test:
  - Image: `Train\safe\c1_safe_driving\img_208.jpg`
  - Prediction: safe (0.9926)

- Unsafe image test:
  - Image: `Train\unsafe\c0_unsafe_driving\img_1844.jpg`
  - Prediction: unsafe (0.9893)

## Conclusion
The TFLite model preserved the classification behavior correctly for both safe and unsafe test images, so it is suitable for the next OpenMV deployment step.