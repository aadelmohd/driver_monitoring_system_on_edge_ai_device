import sensor, time, ml, uos, gc

# ── LED setup ─────────────────────────────────────────────────────────────────
try:
    from pyb import LED
    red_led   = LED(2)  # physically red   on this board
    green_led = LED(1)  # physically green on this board

    def red_on():
        red_led.on()
        green_led.off()

    def green_on():
        red_led.off()
        green_led.on()

    def both_off():
        red_led.off()
        green_led.off()

    def flash_error():
        for _ in range(6):
            red_led.toggle()
            time.sleep_ms(150)
        both_off()

except ImportError:
    from machine import Pin
    red_led   = Pin("LED_GREEN", Pin.OUT)  # swapped to match physical wiring
    green_led = Pin("LED_RED",   Pin.OUT)  # swapped to match physical wiring

    def red_on():
        red_led.high()
        green_led.low()

    def green_on():
        red_led.low()
        green_led.high()

    def both_off():
        red_led.low()
        green_led.low()

    def flash_error():
        for _ in range(6):
            red_led.value(not red_led.value())
            time.sleep_ms(150)
        both_off()

# ── Camera setup ──────────────────────────────────────────────────────────────
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((240, 240))
sensor.set_auto_gain(True)
sensor.set_auto_whitebal(True)
sensor.set_auto_exposure(True, exposure_us=10000)
sensor.skip_frames(time=2000)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_FILE  = "model.tflite"
LABELS_FILE = "labels.txt"

try:
    model_size = uos.stat(MODEL_FILE)[6]
    load_to_fb = model_size <= (gc.mem_free() - (64 * 1024))
    net = ml.Model(MODEL_FILE, load_to_fb=load_to_fb)
except Exception as e:
    flash_error()
    raise Exception("Failed to load " + MODEL_FILE + ": " + str(e))

try:
    labels = [line.rstrip("\n").strip() for line in open(LABELS_FILE)]
except Exception as e:
    flash_error()
    raise Exception("Failed to load " + LABELS_FILE + ": " + str(e))

print("Raw labels:", labels)

# ── Robust label classification ───────────────────────────────────────────────
def label_type(label):
    tokens = []
    for part in label.strip().lower().split():
        tokens.extend(part.split("_"))
    if "unsafe" in tokens:
        return "unsafe"
    if "safe" in tokens:
        return "safe"
    return "unknown"

label_types = [label_type(l) for l in labels]
print("Label types:", list(zip(labels, label_types)))

if "unsafe" not in label_types:
    print("WARNING: No 'unsafe' label found! Check labels.txt")
if "safe" not in label_types:
    print("WARNING: No 'safe' label found! Check labels.txt")

# ── Thresholds & debounce ─────────────────────────────────────────────────────
UNSAFE_THRESHOLD      = 0.70
SAFE_THRESHOLD        = 0.65
UNSAFE_CONFIRM_FRAMES = 3
SAFE_CONFIRM_FRAMES   = 3
GC_INTERVAL_FRAMES    = 30

# ── State ─────────────────────────────────────────────────────────────────────
unsafe_count = 0
safe_count   = 0
alert_on     = False
frame_num    = 0
clock        = time.clock()

# Boot state — green means ready and safe
green_on()

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    clock.tick()
    frame_num += 1

    if frame_num % GC_INTERVAL_FRAMES == 0:
        gc.collect()

    img = sensor.snapshot()

    if frame_num % 2 == 0:
        img.gaussian(1)

    try:
        predictions = net.predict([img])[0].flatten().tolist()
    except Exception as e:
        print("Inference error:", e)
        flash_error()
        time.sleep_ms(200)
        continue

    top_idx   = max(range(len(predictions)), key=lambda i: predictions[i])
    top_score = predictions[top_idx]
    top_label = labels[top_idx]
    top_type  = label_types[top_idx]

    is_unsafe = top_type == "unsafe"
    is_safe   = top_type == "safe"

    # ── Debounce state machine ────────────────────────────────────────────────
    prev_alert = alert_on

    if is_unsafe and top_score >= UNSAFE_THRESHOLD:
        unsafe_count += 1
        safe_count    = 0
        if unsafe_count >= UNSAFE_CONFIRM_FRAMES:
            alert_on = True

    elif is_safe and top_score >= SAFE_THRESHOLD:
        safe_count   += 1
        unsafe_count  = 0
        if safe_count >= SAFE_CONFIRM_FRAMES:
            alert_on = False

    else:
        unsafe_count = max(0, unsafe_count - 1)
        safe_count   = max(0, safe_count   - 1)

    # ── Update LEDs only when state changes ───────────────────────────────────
    if alert_on != prev_alert:
        if alert_on:
            red_on()    # UNSAFE → red LED
        else:
            green_on()  # SAFE   → green LED

    # ── Overlay ───────────────────────────────────────────────────────────────
    if alert_on:
        img.draw_string(5, 5,  "ALERT",  color=(255, 0, 0), scale=2)
        img.draw_string(5, 25, "UNSAFE", color=(255, 0, 0), scale=2)
    else:
        img.draw_string(5, 5,  "SAFE",   color=(0, 255, 0), scale=2)

    print("Pred:", top_label, "Type:", top_type, "Conf:", round(top_score, 2),
          "Alert:", alert_on, "FPS:", round(clock.fps(), 1))
