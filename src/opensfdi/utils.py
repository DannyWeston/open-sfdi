import os
import sys
import cv2
import numpy as np

from contextlib import contextmanager

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

class FringeFactory:
    @staticmethod
    def MakeBinary(frequency, phase_count, orientation, width=1024, height=1024):
        # Maybe not a good idea to rely upon sinusoidal function but works for now :)
        imgs = FringeFactory.MakeSinusoidal(frequency, phase_count, orientation, width, height)

        width, height, _ = imgs[0].shape
        
        for i in range(phase_count):
            for col in range(width):
                for row in range(height):
                    imgs[i][col][row] = 0.0 if imgs[i][col][row] < 0.5 else 1.0

        return imgs

    @staticmethod
    def MakeBinaryRGB(frequency, phase_count, orientation, width=1024, height=1024):
        imgs = FringeFactory.MakeBinary(frequency, phase_count, orientation, width, height)
        
        return FringeFactory.GrayToRGB(imgs)

    @staticmethod
    def MakeSinusoidal(frequency, phase_count, orientation, width=1024, height=1024):
        imgs = np.empty((phase_count, height, width), dtype=np.float32)
        
        for i in range(phase_count):
            x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

            gradient = np.sin(orientation) * x - np.cos(orientation) * y

            imgs[i] = np.sin(((2.0 * np.pi * gradient) / frequency) + i)
            
            imgs[i] = cv2.normalize(imgs[i], None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        
        return imgs

    @staticmethod
    def MakeSinusoidalRGB(frequency, phase_count, orientation, width=1024, height=1024):
        imgs = FringeFactory.MakeSinusoidal(frequency, phase_count, orientation, width, height)
        
        return FringeFactory.GrayToRGB(imgs)

    @staticmethod
    def GrayToRGB(imgs):
        a, b, c = imgs.shape
        rgb_imgs = np.empty(shape=(a, b, c, 3), dtype=imgs[0].dtype)
        
        for i, img in enumerate(imgs):
            rgb_imgs[i] = cv2.merge((img, img, img))

        return rgb_imgs