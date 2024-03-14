
from abc import ABC, abstractmethod

import logging
import cv2

class Projector(ABC):
    def __init__(self, imgs=[], name='Projector1'):
        self.logger = logging.getLogger('sfdi')
        
        self.name = name
        
        self.imgs = imgs
        
        self.img_num = 0

    def display(self):
        img = self.imgs[self.img_num]
        
        self.img_num = (self.img_num + 1) % len(self.imgs)

        return img
    
    def __next__(self):
        temp = self.display()
        
        if temp is None: raise StopIteration
        
        return temp
    
    def __iter__(self):
        return self

class Camera(ABC):
    def __init__(self, resolution=(1280, 720), name='Camera1', cam_mat=None, dist_mat = None, optimal_mat=None):
        self.logger = logging.getLogger('sfdi')
        
        self.resolution = resolution
        
        self.name = name

        self.cam_mat = cam_mat
        self.dist_mat = dist_mat
        self.optimal_mat = optimal_mat
    
    @abstractmethod
    def capture(self):
        pass
    
    def set_resolution(self, res):
        self.resolution = res

    def try_undistort_img(self, img):
        if self.cam_mat is not None and self.dist_mat is not None and self.optimal_mat is not None:
            self.logger.debug('Undistorting camera image...')
            return cv2.undistort(img, self.cam_mat, self.dist_mat, None, self.optimal_mat)
        
        return img

class FakeCamera(Camera):
    def __init__(self, imgs=[], name='Camera1', cam_mat=None, dist_mat = None, optimal_mat=None):
        super().__init__(name='Camera1', cam_mat=cam_mat, dist_mat=dist_mat, optimal_mat=optimal_mat)
        
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
    def __init__(self, img_paths, name='Camera1', cam_mat=None, dist_mat = None, optimal_mat=None):
        super().__init__(name='Camera1', cam_mat=cam_mat, dist_mat=dist_mat, optimal_mat=optimal_mat)
        
        # Load all images into memory
        for path in img_paths:
            self.imgs.append(cv2.imread(path, 1))