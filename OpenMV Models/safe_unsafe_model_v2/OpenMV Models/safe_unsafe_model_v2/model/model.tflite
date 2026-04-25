import sensor, time, ml, uos, gc

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((240, 240))
sensor.skip_frames(time=2000)

try:
    net = ml.Model(
        "trained.tflite",
        load_to_fb=uos.stat("trained.tflite")[6] > (gc.mem_free() - (64 * 1024))
    )
except Exception as e:
    raise Exception("Failed to load trained.tflite: " + str(e))

try:
    labels = [line.rstrip("\n") for line in open("labels.txt")]
except Exception as e:
    raise Exception("Failed to load labels.txt: " + str(e))

print("Labels:", labels)

UNSAFE_THRESHOLD = 0.70
UNSAFE_CONFIRM_FRAMES = 2
SAFE_CONFIRM_FRAMES = 2
PREDICTION_DELAY_MS = 500   # increase to slow down more

unsafe_count = 0
safe_count = 0
alert_on = False

clock = time.clock()

while True:
    clock.tick()

    img = sensor.snapshot()

    predictions = net.predict([img])[0].flatten().tolist()
    predictions_list = list(zip(labels, predictions))

    top_label, top_score = max(predictions_list, key=lambda x: x[1])
    label_lower = top_label.strip().lower()

    is_unsafe = "unsafe" in label_lower
    is_safe = "safe" in label_lower and not is_unsafe

    if is_unsafe and top_score >= UNSAFE_THRESHOLD:
        unsafe_count += 1
        safe_count = 0

        if unsafe_count >= UNSAFE_CONFIRM_FRAMES:
            alert_on = True

    elif is_safe:
        safe_count += 1
        unsafe_count = 0

        if safe_count >= SAFE_CONFIRM_FRAMES:
            alert_on = False

    if alert_on:
        img.draw_string(5, 5, "ALERT", color=(255, 0, 0), scale=2)
        img.draw_string(5, 25, "UNSAFE", color=(255, 0, 0), scale=2)
    else:
        img.draw_string(5, 5, "SAFE", color=(0, 255, 0), scale=2)

    print("Pred:", top_label, "Conf:", round(top_score, 2),
          "Alert:", alert_on, "FPS:", round(clock.fps(), 1))

    time.sleep_ms(PREDICTION_DELAY_MS)
