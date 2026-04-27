# ============================================================
# Speed Limit Sign Detection - OpenMV Project
# Driver Monitoring System on Edge AI Device
# ============================================================
# Compatible with: OpenMV H7 / H7 Plus / RT
# Firmware:        OpenMV MicroPython >= 4.x
#
# Approach:
#   1. Detect the red circular border of speed limit signs
#      using colour blob detection (LAB thresholds).
#   2. Inside each candidate blob ROI crop, look for white
#      centre area (the digit background).
#   3. Use a tiny Edge Impulse / TFLite classification model
#      (speed_limit_classifier.tflite) dropped into the
#      OpenMV storage to read the digit (30 / 50 / 60 / 80 /
#      100 / 120 km/h).  If no model is present the script
#      falls back to showing the raw detection box.
#   4. Overlay the result on screen and optionally send a
#      UART alert when a limit is detected.
#
# Files needed on OpenMV storage (/):
#   speed_limit_detection.py   <- this file
#   speed_limit_classifier.tflite  (optional but recommended)
#   speed_limit_labels.txt         (optional, one label/line)
# ============================================================

import sensor
import image
import time
import math

# ── Optional: UART alert output ──────────────────────────────
try:
    from pyb import UART
    uart = UART(3, 9600, timeout_char=1000)
    UART_ENABLED = True
except Exception:
    UART_ENABLED = False

# ── Optional: TFLite model ───────────────────────────────────
MODEL_PATH   = "speed_limit_classifier.tflite"
LABELS_PATH  = "speed_limit_labels.txt"
USE_MODEL    = False
net          = None
labels       = []

try:
    import tf
    net    = tf.load(MODEL_PATH, load_to_fb=True)
    USE_MODEL = True
    try:
        with open(LABELS_PATH) as f:
            labels = [l.rstrip("\n") for l in f.readlines()]
    except Exception:
        labels = ["30","40","50","60","70","80","100","120"]
    print("[INFO] TFLite model loaded:", MODEL_PATH)
except Exception as e:
    print("[WARN] No TFLite model found, running blob-only mode:", e)

# ── Camera setup ─────────────────────────────────────────────
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)          # 320×240 – fast enough
sensor.set_vflip(False)
sensor.set_hmirror(False)
sensor.skip_frames(time=2000)              # warm-up
sensor.set_auto_gain(True)
sensor.set_auto_whitebal(True)
clock = time.clock()

# ── Colour thresholds (LAB colour space) ────────────────────
# Red hue of EU/UK speed-limit circle border.
# Tune L / A / B min/max for your lighting conditions.
RED_THRESHOLD = (20, 70,   # L  (luminance)
                 25, 80,   # A  (green→red)
                -20, 30)   # B  (blue→yellow)

WHITE_THRESHOLD = (70, 100,   # L
                   -15, 15,   # A
                   -15, 15)   # B

# ── Detection parameters ────────────────────────────────────
MIN_BLOB_AREA     = 400     # px² – ignore tiny blobs
MAX_BLOB_AREA     = 30000   # px² – ignore huge false positives
CIRCULARITY_MIN   = 0.55    # 0..1  – real signs are roundish
WHITE_FILL_RATIO  = 0.25    # centre must be ≥ 25 % white

# ── Utilities ───────────────────────────────────────────────
def circularity(blob):
    """4π·Area / Perimeter²  → 1.0 for a perfect circle."""
    p = blob.perimeter()
    if p == 0:
        return 0
    return (4 * math.pi * blob.pixels()) / (p * p)

def white_ratio(img, roi):
    """Fraction of white pixels inside roi."""
    crop   = img.copy(roi=roi)
    blobs  = crop.find_blobs([WHITE_THRESHOLD],
                              pixels_threshold=10,
                              area_threshold=10,
                              merge=True)
    if not blobs:
        return 0.0
    white_px = sum(b.pixels() for b in blobs)
    total_px = roi[2] * roi[3]
    return white_px / total_px if total_px > 0 else 0.0

def classify_roi(img, roi):
    """Run TFLite model on the ROI and return (label, confidence)."""
    if not USE_MODEL or net is None:
        return ("?", 0.0)
    try:
        obj = tf.classify(net, img, roi=roi,
                          min_scale=1.0, scale_mul=0.5,
                          x_overlap=0.0, y_overlap=0.0)
        if obj:
            top = max(obj[0].output(), key=lambda x: x)
            idx = list(obj[0].output()).index(top)
            label = labels[idx] if idx < len(labels) else str(idx)
            return (label, top)
    except Exception as e:
        print("[WARN] classify error:", e)
    return ("?", 0.0)

def send_uart(label, confidence):
    if UART_ENABLED and label != "?":
        msg = "SPEED_LIMIT:{} CONF:{:.0f}%\r\n".format(label, confidence * 100)
        uart.write(msg)

# ── Drawing helpers ─────────────────────────────────────────
COLORS = {
    "detected": (255, 80,  0),    # orange box
    "label":    (255, 255, 0),    # yellow text
    "fps":      (0,   200, 255),  # cyan
}

def draw_overlay(img, blob, label, conf):
    x, y, w, h = blob.rect()
    img.draw_rectangle(x, y, w, h, color=COLORS["detected"], thickness=2)
    img.draw_circle(blob.cx(), blob.cy(), max(w, h)//2,
                    color=COLORS["detected"], thickness=1)
    text = "{} {:.0f}%".format(label, conf * 100) if label != "?" else "Sign?"
    img.draw_string(x, y - 12 if y > 12 else y + h + 2,
                    text, color=COLORS["label"], scale=1.4)

# ── Main loop ───────────────────────────────────────────────
print("[INFO] Speed limit detection started.")

last_label    = "None"
last_conf     = 0.0
alert_counter = 0     # hold alert on screen for N frames

while True:
    clock.tick()
    img = sensor.snapshot()

    # ── 1. Find red blobs ───────────────────────────────────
    red_blobs = img.find_blobs(
        [RED_THRESHOLD],
        pixels_threshold=MIN_BLOB_AREA // 4,
        area_threshold=MIN_BLOB_AREA,
        merge=True,
        margin=5,
    )

    detected_this_frame = False

    for blob in red_blobs:
        area = blob.pixels()
        if area > MAX_BLOB_AREA:
            continue

        circ = circularity(blob)
        if circ < CIRCULARITY_MIN:
            continue

        # ── 2. Check for white centre (digit area) ──────────
        bx, by, bw, bh = blob.rect()
        # Shrink to inner 60 % for white check
        inner_margin_x = int(bw * 0.20)
        inner_margin_y = int(bh * 0.20)
        inner_roi = (
            bx + inner_margin_x,
            by + inner_margin_y,
            max(bw - 2 * inner_margin_x, 1),
            max(bh - 2 * inner_margin_y, 1),
        )

        # Clamp ROI to image bounds
        iw, ih = img.width(), img.height()
        inner_roi = (
            max(0, inner_roi[0]),
            max(0, inner_roi[1]),
            min(inner_roi[2], iw - inner_roi[0]),
            min(inner_roi[3], ih - inner_roi[1]),
        )

        if inner_roi[2] < 5 or inner_roi[3] < 5:
            continue

        wr = white_ratio(img, inner_roi)
        if wr < WHITE_FILL_RATIO:
            continue

        # ── 3. Classify the sign ────────────────────────────
        label, conf = classify_roi(img, blob.rect())

        # ── 4. Draw and alert ────────────────────────────────
        draw_overlay(img, blob, label, conf)
        detected_this_frame = True

        if label != last_label or conf > last_conf:
            last_label    = label
            last_conf     = conf
            alert_counter = 30          # show alert for 30 frames
            send_uart(label, conf)

    # ── Persistent alert banner ─────────────────────────────
    if alert_counter > 0:
        alert_counter -= 1
        banner = "Speed limit: {} km/h".format(last_label) \
                 if last_label not in ("None", "?") \
                 else "Speed sign detected"
        img.draw_rectangle(0, 0, img.width(), 18, color=(0, 0, 0), fill=True)
        img.draw_string(4, 2, banner, color=COLORS["label"], scale=1.2)

    # ── FPS counter ─────────────────────────────────────────
    fps_str = "FPS: {:.1f}".format(clock.fps())
    img.draw_string(img.width() - 72, img.height() - 14,
                    fps_str, color=COLORS["fps"], scale=1.0)

    # ── Debug output every 30 frames ────────────────────────
    if int(clock.fps()) % 30 == 0:
        print("[DEBUG] FPS={:.1f}  last={}  conf={:.0f}%".format(
              clock.fps(), last_label, last_conf * 100))
