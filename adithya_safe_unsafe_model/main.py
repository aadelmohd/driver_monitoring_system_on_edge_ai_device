import sensor
import time

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)

clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()
    img.draw_string(10, 10, "Camera OK", color=(255, 0, 0), scale=2)
    print("FPS:", clock.fps())
