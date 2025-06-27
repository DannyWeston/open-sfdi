import cv2

from ..image import Show

def show_phasemap(phasemap, name='Phasemap'):
    # Mark nans as black

    norm = cv2.normalize(phasemap, None, 0.0, 1.0, cv2.NORM_MINMAX)
    Show(norm, name)