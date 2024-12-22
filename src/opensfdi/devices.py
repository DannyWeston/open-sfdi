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

# class MotorStage(Protocol):
#     @property
#     def min_height(self) -> float:
#         pass

#     @property
#     def max_height(self) -> float:
#         pass
    
#     def move_to(self, value):
#         raise NotImplementedError