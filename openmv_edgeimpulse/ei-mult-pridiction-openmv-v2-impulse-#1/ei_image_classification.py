import sensor, time, ml, uos, gc

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((240, 240))
sensor.skip_frames(time=2000)

# Try to load LED gracefully
try:
    from machine import LED
    red_led = LED("LED_RED")
    has_led = True
except:
    try:
        import pyb
        red_led = pyb.LED(1)
        has_led = True
    except:
        has_led = False
        print("No LED support on this board")

net = None
labels = None

try:
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite" (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt" (' + str(e) + ')')

clock = time.clock()

# --- Non-blocking LED state ---
led_state     = False
last_led_time = 0
LED_INTERVAL  = 200  # ms between blinks — lower = faster blink

def update_led(is_unsafe):
    global led_state, last_led_time
    if not has_led:
        return
    if is_unsafe:
        now = time.ticks_ms()
        if time.ticks_diff(now, last_led_time) >= LED_INTERVAL:
            led_state = not led_state          # toggle
            if led_state:
                red_led.on()
            else:
                red_led.off()
            last_led_time = now
    else:
        red_led.off()
        led_state = False

def clean_label(label):
    return label.replace("unsafe(", "").replace(")", "").upper()

while True:
    clock.tick()
    img = sensor.snapshot()

    predictions = list(zip(labels, net.predict([img])[0].flatten().tolist()))
    best_label, best_score = max(predictions, key=lambda x: x[1])

    is_unsafe = best_label != "safe"
    update_led(is_unsafe)   # non-blocking — no sleep!

    if not is_unsafe:
        print("Status: SAFE (%.1f%%)" % (best_score * 100))
    else:
        reason = clean_label(best_label)
        print("Status: UNSAFE - %s (%.1f%%)" % (reason, best_score * 100))

    print("%.1f fps" % clock.fps())
