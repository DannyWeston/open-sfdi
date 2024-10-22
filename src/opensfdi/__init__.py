from skimage.restoration import unwrap_phase

import numpy as np
import cv2
from matplotlib import pyplot as plt

import logging
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true')

args, unknown = parser.parse_known_args()

args = vars(args)

DEBUG = args["debug"]

logger = logging.getLogger(__name__)

#formatter = logging.Formatter(fmt='%(threadName)s:%(message)s')
formatter = logging.Formatter(fmt='[%(levelname)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

def show_phasemap(phasemap, min_phase=None, max_phase=None):
    plt.imshow(phasemap, cmap='gray', vmin=min_phase, vmax=max_phase)
    plt.title(f'Phasemap ({phasemap.min():.2f} to {phasemap.max():.2f})')
    plt.show()

def show_surface(data):
    hf = plt.figure()

    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))

    ha.plot_surface(X, Y, data)

    plt.show()

def show_image(img, title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def centre_crop_img(img, x1, y1, x2:int = 0, y2:int = 0):
    if x2 == 0:
        x2 = img.shape[1] if x1 == 0 else -x1
        
    if y2 == 0:
        y2 = img.shape[0] if y1 == 0 else -y1
    
    return img[y1 : y2, x1 : x2]

def normalise_image(img):
    return ((img - img.min()) / (img.max() - img.min()))

def rgb2grey(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def wrapped_phase(imgs):
    h, w = imgs[0].shape
    N = len(imgs)

    p = np.zeros(shape=(h, w), dtype=np.float64)
    q = np.zeros(shape=(h, w), dtype=np.float64)

    # Accumulate phases
    for i, img in enumerate(imgs):
        phase = (2.0 * np.pi * i) / N

        p += img * np.sin(phase)
        q += img * np.cos(phase)

    return -np.arctan2(p, q)

def ac_imgs(imgs: list):
    return np.divide(np.sum(imgs, axis=0), len(imgs))

def dc_imgs(imgs: list):
    N = len(imgs)
    
    p = q = np.zeros(imgs[0].shape, dtype=np.float32)
    
    for i, img in enumerate(imgs):
        phase = (2.0 * np.pi * i) / N
        
        p = np.add(p, img * np.sin(phase))
        q = np.add(q, img * np.cos(phase))
        
    return (2.0 / N) * np.sqrt((p * p) + (q * q))