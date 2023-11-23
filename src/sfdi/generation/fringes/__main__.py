import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

import cv2
import time

# LG Projector spatial frequency - pixels per cm
# e.g: if the projected image is 10 cm wide for 1280 pixels, then 128 pixels wide per cm

LG_PROTECTOR_SF = 128
def sinusoidal(width, height, freq, phase=0, orientation=np.pi / 2):
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    gradient = np.sin(orientation) * x - np.cos(orientation) * y
    return np.sin((2 * np.pi * gradient) / freq + phase)

def phase_animation(width, height, freq, orientation):
    counter = 0

    while True:
        img = sinusoidal(width, height, freq, counter, orientation)
        img = cv2.resize(img, (int(width / 3), int(height / 3)))  

        cv2.imshow('Phase Shifting', img)
        key = cv2.waitKey(int(1000 / 30))
        counter += 0.05
        if key == 27:
            cv2.destroyAllWindows()
            break

def generate_images(width, height, freq, orientation, n=3):
    return [sinusoidal(width, height, freq, 2 * i * np.pi / n, orientation) for i in range(n)]

width = 1280
height = 720

freq = LG_PROTECTOR_SF
orientation = np.pi / 2
imgs = generate_images(width, height, freq, orientation)

phase_animation(width, height, freq, orientation)