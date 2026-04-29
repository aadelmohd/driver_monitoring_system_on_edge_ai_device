# My Contribution: Safe vs Unsafe Driver Monitoring Model

## OpenMV setup
- Connected OpenMV camera to laptop
- Verified live camera feed using `main.py`
- Saved `main.py` to the OpenMV board for camera sanity check

## Model training work
- Trained a binary image classification model for safe vs unsafe driver behavior
- Dataset size:
  - Safe: 1020 images
  - Unsafe: 1200 images
- Total: 2220 images
- Used stratified 80/20 train-validation split
- Built and evaluated a CNN model in TensorFlow

## Final validation performance
- Validation accuracy: 90.32%

## Classification report
- Safe: precision 0.90, recall 0.88, f1-score 0.89
- Unsafe: precision 0.90, recall 0.92, f1-score 0.91

## Confusion matrix
[[180, 24],
 [19, 221]]

## Files added
- `main.py`
- `scripts/train_safe_unsafe.py`
- `scripts/predict_image.py`
- `scripts/check_folder_predictions.py`
- `models/safe_unsafe_model.keras`
- `results/training_curves.png`
- `notes/model_results.md`

## Unseen Test Set Evaluation
A separate test set was also used to check final model performance.

### Test Accuracy
- **91.67%**

### Test Classification Report
- **Safe**
  - Precision: 1.00
  - Recall: 0.83
  - F1-score: 0.91

- **Unsafe**
  - Precision: 0.86
  - Recall: 1.00
  - F1-score: 0.92

### Test Confusion Matrix
[[10, 2],
 [0, 12]]