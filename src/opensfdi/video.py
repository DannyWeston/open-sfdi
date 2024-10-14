from typing import Protocol

class FringeProjector(Protocol):
    @property
    def frequency(self) -> int:
        pass

    @property
    def resolution(self) -> tuple[int, int]:
        pass
        
    @property
    def rotation(self) -> float:
        pass
    
    @property
    def phase(self) -> float:
        pass

    def display(self):
        """Display the fringes"""

    # @property
    # def phases(self):
    #     return self.__phases

    # @phases.setter
    # def phases(self, value):
    #     self.__phases = value
    #     self.current_phase = 0

    # @property
    # def current_phase(self):
    #     return None if len(self.phases) == 0 else self.phases[self.__current]
    
    # @current_phase.setter
    # def current_phase(self, value):
    #     self.__current = value

    # def next_phase(self):
    #     self.current_phase = (self.current_phase + 1) % len(self.phases)

class Camera(Protocol):

    @property
    def resolution(self) -> tuple[int, int]:
        pass

    @property
    def distortion(self) -> object:
        pass
    
    def capture(self):
        """Capture an image"""

    # def try_undistort_img(self, img):
    #     if self.cam_mat is not None and self.dist_mat is not None and self.optimal_mat is not None:
    #         self.logger.debug('Undistorting camera image...')
    #         return cv2.undistort(img, self.cam_mat, self.dist_mat, None, self.optimal_mat)
        
    #     return img

# class FakeCamera(Camera):
#     def __init__(self, imgs=[], name='Camera1', cam_mat=None, dist_mat = None, optimal_mat=None):
#         super().__init__(name='Camera1', cam_mat=cam_mat, dist_mat=dist_mat, optimal_mat=optimal_mat)
        
#         self.img_num = 0

#         self.imgs = imgs

#     def capture(self):
#         img = next(self.imgs)
        
#         if not self.loop and len(self.imgs) <= self.img_num:
#             self.img_num = 0
#             return None
        
#         self.img_num += 1
        
#         self.logger.info(f'Returning an image')

#         return img
    
#     def __iter__(self):
#         return iter(self.imgs)

#     def add_image(self, img):
#         self._images.append(img)
#         return self

# class FileCamera(FakeCamera):
    # def __init__(self, img_paths, name='Camera1', cam_mat=None, dist_mat = None, optimal_mat=None):
    #     super().__init__(name='Camera1', cam_mat=cam_mat, dist_mat=dist_mat, optimal_mat=optimal_mat)
        
    #     # Load all images into memory
    #     for path in img_paths:
    #         self.imgs.append(cv2.imread(path, 1))