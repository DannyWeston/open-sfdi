import cv2

import numpy as np


from sfdi.definitions import FRINGES_DIR

class FringeFactory:
    @staticmethod
    def MakeBinary(frequency, phase_count, orientation, width=1024, height=1024):
        # Maybe not a good idea to rely upon sinusoidal function but works for now :)
        imgs = FringeGenerator.MakeSinusoidal(frequency, phase_count, orientation, rgba, width, height)

        width, height, _ = imgs[0].shape
        
        for i in range(phase_count):
            for col in range(width):
                for row in range(height):
                    imgs[i][col][row] = 0.0 if imgs[i][col][row] < 0.5 else 1.0

        return imgs

    @staticmethod
    def MakeBinaryRGB(frequency, phase_count, orientation, width=1024, height=1024):
        imgs = FringeFactory.MakeBinary(frequency, phase_count, orientation, width, height)
        
        return FringeGenerator.GrayToRGB(imgs)

    @staticmethod
    def MakeSinusoidal(frequency, phase_count, orientation, width=1024, height=1024):
        imgs = np.empty((phase_count, width, height))
        
        for i in range(phase_count):
            x, y = np.meshgrid(np.arange(width, dtype=int), np.arange(height, dtype=int))

            gradient = np.sin(orientation) * x - np.cos(orientation) * y

            imgs[i] = np.sin(((2.0 * np.pi * gradient) / freq) + phase)
            
            imgs[i] = cv2.normalize(img[i], None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        
        return imgs

    @staticmethod
    def MakeSinusoidalRGB(frequency, phase_count, orientation, width=1024, height=1024):
        imgs = FringeFactory.MakeSinusoidal(frequency, phase_count, orientation, width, height)
        
        return FringeGenerator.GrayToRGB(imgs)

    @staticmethod
    def GrayToRGB(imgs):
        count, width, height = imgs.shape
        
        rgb_imgs = np.empty((count, width, height, 3))
        
        for i in range(phase_count):
           rgb_imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2RGB)
           
        return rgb_imgs