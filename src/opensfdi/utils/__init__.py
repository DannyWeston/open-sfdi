import cv2
import os
import sys

import numpy as np


# Redirect stdout to /dev/null
@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)

def centre_crop_img(img, x1, y1, x2:int = 0, y2:int = 0):
    if x2 == 0:
        x2 = img.shape[1] if x1 == 0 else -x1
        
    if y2 == 0:
        y2 = img.shape[0] if y1 == 0 else -y1
    
    return img[y1 : y2, x1 : x2]


def mask_heightmap(self, heightmap):
    """ Does something

    Args:
        heightmap (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Smooth over the heightmap with median filter
    smoothed = cv2.medianBlur(heightmap, 3)

    smoothed = cv2.normalize(smoothed, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    # Find Canny edges
    edged = cv2.Canny(smoothed, 30, 200) 
    cv2.imshow('Canny Edges After Contouring', edged)
    cv2.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    print("Number of Contours found = " + str(len(contours))) 
    
    # Draw all contours 
    # -1 signifies drawing all contours 
    cv2.drawContours(smoothed, contours, -1, (0, 255, 0), 3) 
    
    cv2.imshow('Contours', smoothed) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return smoothed


def normalise_img(img):
    return ((img - img.min()) / (img.max() - img.min()))