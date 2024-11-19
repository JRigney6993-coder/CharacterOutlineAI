import mss
import numpy as np

def capture_screen(region=None):
    with mss.mss() as sct:
        monitor = region or sct.monitors[1]
        screen = sct.grab(monitor)
        return np.array(screen)