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

logger = logging.getLogger('opensfdi')

#formatter = logging.Formatter(fmt='%(threadName)s:%(message)s')
formatter = logging.Formatter(fmt='[%(levelname)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

def show_surface(data):
    hf = plt.figure()

    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(range(len(data[0])), range(len(data)))  # `plot_surface` expects `x` and `y` data to be 2D
    temp = np.mean(data, axis=2) if data.ndim == 3 else data
    ha.plot_surface(X, Y, temp)

    plt.show()

def display_image(img, grey=False, title='', vmin=0.0, vmax=1.0):
    if grey:
        cmap='gray'
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        cmap='jet'

    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.show()

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

def unwrapped_phase(phi_imgs):
    return unwrap_phase(phi_imgs)



def unwrap_itoh_1d(wrapped):
    k = 0

    addon = np.zeros(wrapped.shape)

    for i in range(0, len(wrapped) - 1):
        difference = wrapped[i + 1] - wrapped[i]
        if np.pi < difference:
            k -= (2.0 * np.pi)

        elif difference < -np.pi:
            k += (2.0 * np.pi)

        addon[i] = k

    return wrapped + addon

def unwrap_itoh_2d(wrapped):
    # TODO: Maybe possible to do this in place?

    a, b = wrapped.shape

    unwrapped = np.zeros((a, b))
    for i in range(0, a):
        unwrapped[i, :] = unwrap_itoh_1d(wrapped[i, :])

    for j in range(0, b):
        unwrapped[:, j] = unwrap_itoh_1d(wrapped[:, j])

    return unwrapped

def unwrap_reliability_2d(wrapped):
    
    
    return None


def wrapped_phase(imgs):
    p = np.zeros(imgs[0].shape, dtype=np.float32)
    q = np.zeros(imgs[0].shape, dtype=np.float32)

    for i, img in enumerate(imgs):
        phase = (2.0  * np.pi * i) / len(imgs)
        p = np.add(p, img * np.sin(phase), dtype=np.float32)
        q = np.add(q, img * np.cos(phase), dtype=np.float32)

    return np.negative(np.arctan2(p, q), dtype=np.float32)

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