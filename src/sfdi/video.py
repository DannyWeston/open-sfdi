
from abc import ABC, abstractmethod

import logging
import cv2

class Projector(ABC):
    def __init__(self, imgs=[]):
        self.logger = logging.getLogger('sfdi')
        
        self.imgs = imgs
        
        self.img_num = 0

    def display(self):
        img = self.imgs[self.img_num]
        
        self.img_num = (self.img_num + 1) % len(self.imgs)

        return img
    
    def __iter__(self):
        return iter(self.imgs)

class Camera(ABC):
    def __init__(self, resolution=(1280, 720)):
        self.logger = logging.getLogger('sfdi')
        
        self.resolution = resolution
    
    @abstractmethod
    def capture(self):
        pass
    
    def set_resolution(self, res):
        self.resolution = res

class FakeCamera(Camera):
    def __init__(self, imgs=[]):
        super().__init__()
        
        self.img_num = 0

        self.imgs = imgs

    def capture(self):
        img = next(self.imgs)
        
        if not self.loop and len(self.imgs) <= self.img_num:
            self.img_num = 0
            return None
        
        self.img_num += 1
        
        self.logger.info(f'Returning an image')

        return img
    
    def __iter__(self):
        return iter(self.imgs)

    def add_image(self, img):
        self._images.append(img)
        return self

class FileCamera(FakeCamera):
    def __init__(self, img_paths):
        super().__init__()
        
        # Load all images into memory
        for path in img_paths:
            self.imgs.append(cv2.imread(path, 1))