# Safe vs Unsafe Driving Detection (v2)

## Overview

This project detects **safe vs unsafe driving behavior** using an Edge Impulse model deployed on OpenMV.

It runs in real-time and shows:

* **SAFE** (green)
* **UNSAFE + ALERT** (red)

---

## Structure

```
safe_unsafe_model_v2/
  model/
    model.tflite
    labels.txt
    inference.py

  metrics/
    confusion_matrix.png
    metrics_summary.png
    edge_impulse_results.png

  recordings/
    demo_run.mp4
```

---

## Features

* Real-time detection on OpenMV
* Confidence threshold for unsafe detection
* Multiple frame check to reduce false alerts

---

## Performance

* Accuracy: ~98%
* F1 Score: ~0.99

---

## Demo

See `recordings/demo_run.mp4`

---

## Notes

* Model trained using Edge Impulse
* Optimized for edge device (OpenMV)

---

## Author

Tirthraj
