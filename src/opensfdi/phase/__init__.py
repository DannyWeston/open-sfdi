import cv2

from ..image import Show
from ..utils import AlwaysNumpy

def ShowPhasemap(phasemap, name='Phasemap', size=None):
    # Mark nans as black

    phasemap = AlwaysNumpy(phasemap)

    norm = cv2.normalize(phasemap, None, 0.0, 1.0, cv2.NORM_MINMAX)
    Show(norm, name, size=size)