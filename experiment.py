
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy import ndimage, misc
import pandas as pd

# Demodulation (array input)
def AC(var: list):
    return ((2) ** (1 / 2) / 3) * (((var[0] - var[1]) ** 2 + (var[1] - var[2]) ** 2 + (var[2] - var[0]) ** 2) ** (1 / 2))

def DC(var: list):
    return (1 / 3) * (var[0] + var[1] + var[2])

class Projection:
    def __init__(self, proj_imgs, img_func = None):
        self.proj_imgs = proj_imgs
        self.img_func = img_func

    def run(self):
        imgs = []
        ref_imgs = []

        # Try and load the projector images, 3 phrases
        for proj_img in self.proj_imgs:
            img = self.__load_img(proj_img)

            if self.img_func: img = self.img_func(img) # Apply some filtering to image if valid
            
            imgs.append(img)
            ref_imgs.append(img)

    def __load_img(self, path):
        img = cv2.imread(path, 1)

        self.__display_img(img)
        cv2.destroyAllWindows()

        return img.astype(np.double)
    
    def __display_img(self, img):
        cv2.namedWindow("main", cv2.WND_PROP_FULLSCREEN)          
        cv2.setWindowProperty("main", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("main", img)
        cv2.waitKey(0)