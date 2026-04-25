# Safe vs Unsafe Driving Detection (v2)

## Overview

This project detects **safe vs unsafe driving behavior** using an Edge Impulse model on an OpenMV device.

---

## Project Structure

```text
safe_unsafe_model_v2/
  model/        -> model files (tflite, labels, code)
  metrics/      -> evaluation results (confusion matrix, metrics)
  recordings/   -> demo video output
  README.md
```

---

## How it works

* Captures image from camera
* Runs ML model
* Classifies as SAFE or UNSAFE
* Shows alert for unsafe driving

---

## Features

* Real-time detection
* Confidence threshold filtering
* Multi-frame confirmation (reduces false alerts)

---

## Performance

* Accuracy: ~98.6%
* F1 Score: ~0.99

---

## Notes

* Model trained using Edge Impulse
* Runs on OpenMV (edge device)

---

## Author

Tirthraj
