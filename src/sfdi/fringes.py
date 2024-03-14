import cv2
import numpy as np

from sfdi.definitions import FRINGES_DIR

class Fringes:
    def binary(width, height, freq, orientation, phase, rgba=True):
        # Maybe not a good idea to rely upon sinusoidal function
        # But works for now :)

        img = Fringes.sinusoidal(width, height, freq, phase, orientation)
        width, height = img.shape
        if rgba:
            for col in range(width):
                for row in range(height):
                    img[col][row][:] = 0.0 if img[col][row][0] < 0.5 else 1.0
        else:
            for col in range(width):
                for row in range(height):
                    img[col][row] = 0.0 if img[col][row] < 0.5 else 1.0

        return img

    def sinusoidal(width, height, freq, orientation, phase, rgba=True):
        x, y = np.meshgrid(np.arange(width, dtype=int), np.arange(height, dtype=int))

        gradient = np.sin(orientation) * x - np.cos(orientation) * y
        
        img = np.sin(((2.0 * np.pi * gradient) / freq) + phase)
        
        img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        
        if rgba: return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def __init__(self, fringe_imgs):
        self._images = fringe_imgs

    def __iter__(self):
        return iter(self._images)

    def __next__(self):
        return next(self._images)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        return self._images[item]

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        self._images = images

    def from_generator(width, height, freq, orientation=(np.pi / 2.0), n=3, fringe_type='Sinusoidal'):
        if fringe_type == 'Sinusoidal': gen_func = Fringes.sinusoidal
        elif fringe_type == 'Binary': gen_func = Fringes.binary
        else: raise Exception("Incorrect fringe generator function provided")

        imgs = np.array([gen_func(width, height, freq, orientation, (2.0 * i * np.pi) / n) for i in range(n)])

        return Fringes(imgs)