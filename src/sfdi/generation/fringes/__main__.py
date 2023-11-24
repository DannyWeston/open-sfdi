import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

import cv2
import time

import os

from sfdi.definitions import FRINGES_DIR

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
        key = cv2.waitKey(int(1000 / 30)) # Roughly 30 fps
        counter += 0.05
        if key == 27:
            cv2.destroyAllWindows()
            break

def generate_images(width, height, freq, orientation, n=3):
    imgs = []

    for i in range(n):
        img = sinusoidal(width, height, freq, 2 * i * np.pi / n, orientation)
        img = ((img - img.min()) / (img.max() - img.min())) * 255
        imgs.append(img)

    return imgs

def save_image(img, name):
    out = os.path.join(FRINGES_DIR, name)
    cv2.imwrite(out, img)

# LG Projector spatial frequency - pixels per cm
# e.g: if the projected image is 10 cm wide for 1280 pixels, then 128 pixels wide per cm
projector_sfs = {
    'LG' : 32,
}

width = 1280
height = 720

freq = projector_sfs['LG']
orientation = np.pi / 2
imgs = generate_images(width, height, freq, orientation)

for i, img in enumerate(imgs):
    save_image(img, f'fringes_{i}.jpg')

#phase_animation(width, height, freq, orientation)