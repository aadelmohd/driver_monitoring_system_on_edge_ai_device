import sensor
import time
import ml
from ml.preprocessing import Normalization

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)

norm = Normalization(scale=(0.0, 1.0))
net = ml.Model("safe_unsafe_model.tflite")

clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()

    output = net.predict([norm(img)])[0].flatten()[0]
    unsafe_score = float(output)

    if unsafe_score >= 0.5:
        label = "unsafe"
        score = unsafe_score
        color = (255, 0, 0)
    else:
        label = "safe"
        score = 1.0 - unsafe_score
        color = (0, 255, 0)

    img.draw_string(5, 5, "{} {:.2f}".format(label, score), color=color, scale=2)
    print("label:", label, "unsafe_score:", unsafe_score, "fps:", clock.fps())
