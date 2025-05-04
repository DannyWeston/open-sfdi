import numpy as np
import cv2

from abc import ABC, abstractmethod

from . import image

class Camera(ABC):
    @abstractmethod
    def __init__(self, resolution=(720, 1280), channels=3):
        self.__proj_mat = None
        self.__dist_mat = None

        self.__channels = channels
        self.__resolution = resolution

    @property
    def dist_mat(self) -> np.ndarray:
        return self.__dist_mat

    @dist_mat.setter
    def dist_mat(self, value):
        self.__dist_mat = value

    @property
    def proj_mat(self) -> np.ndarray:
        return self.__proj_mat

    @proj_mat.setter
    def proj_mat(self, value):
        self.__proj_mat = value

    @property
    def channels(self) -> int:
        return self.__channels
    
    @channels.setter
    def channels(self, value) -> int:
        if value < 1: return # Silently do nothing (maybe change to Exception?)\
        
        self.__channels = value

    @property
    def shape(self) -> tuple:
        if self.channels == 1:
            return self.resolution

        return (*self.resolution, self.channels)

    @property
    def resolution(self) -> tuple[int, int]:
        return self.__resolution
    
    @resolution.setter
    def resolution(self, value):
        self.__resolution = value
        
    def is_calibrated(self) -> bool:
        return not (self.proj_mat is None)

    def calibrate(self, world_xyz, corner_pixels):
        E, K, self.__dist_mat, R, t = cv2.calibrateCamera(world_xyz, corner_pixels, self.resolution, None, None)

        R, _  = cv2.Rodrigues(R[0])
        t = t[0]

        # Calculate camera projection matrix (use first entry, could use any)
        self.proj_mat = np.dot(K, np.hstack((R, t)))

        return E

    @abstractmethod
    def capture(self) -> image.Image:
        """ Capture an image NOTE: should return pixels with float32 type """

    # def try_undistort_img(self, img):
    #     if self.cam_mat is not None and self.dist_mat is not None and self.optimal_mat is not None:
    #         self.logger.debug('Undistorting camera image...')
    #         return cv2.undistort(img, self.cam_mat, self.dist_mat, None, self.optimal_mat)
        
    #     return img

class Projector(ABC):
    @abstractmethod
    def __init__(self):
        self.__proj_mat = None
        self.__dist_mat = None

    def is_calibrated(self) -> bool:
        return not (self.proj_mat is None)

    @property
    def dist_mat(self) -> np.ndarray:
        return self.__dist_mat
    
    @dist_mat.setter
    def dist_mat(self, value):
        self.__dist_mat = value

    @property
    def proj_mat(self) -> np.ndarray:
        return self.__proj_mat
    
    @proj_mat.setter
    def proj_mat(self, value):
        self.__proj_mat = value

    def calibrate(self, w_xyz, corner_pixels, phasemaps, num_stripes):
        # TODO: Investigate optimisation as mapping error is introduced by using subpixel values
        # Convert camera corner pixels to ints for indexing
        corner_pixels = np.round(corner_pixels).astype(int)

        h, w = self.resolution
        f_v = w / num_stripes # Vertical fringe frequency
        f_h = h / num_stripes # Horizontal fringe frequency

        proj_coords = np.empty_like(corner_pixels, dtype=np.float32)

        for i in range(0, len(phasemaps), 2):
            c_xs = corner_pixels[i,:,0] # Vertical fringes corner pixels
            c_ys = corner_pixels[i,:,1] # Horizontal fringes corner pixels

            phi_v = phasemaps[i]    # Vertical fringes phasemap
            phi_h = phasemaps[i+1]  # Horizontal fringes phasemap

            # Convert camera coordinates to projector coordinates
            # "act as the projector's eye"
            proj_xs = (phi_v[c_ys, c_xs] * f_v) / (2.0 * np.pi)
            proj_ys = (phi_h[c_ys, c_xs] * f_h) / (2.0 * np.pi)

            proj_coords[i] = np.dstack((proj_xs, proj_ys))
            proj_coords[i+1] = np.dstack((proj_xs, proj_ys))

        E, K, self.__dist_mat, R, t = cv2.calibrateCamera(w_xyz, proj_coords, self.resolution, None, None)

        R, _ = cv2.Rodrigues(R[0])
        t = t[0]

        self.proj_mat = np.dot(K, np.hstack((R, t)))

        return E
    

    # Abstract methods

    @property
    @abstractmethod
    def frequency(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        raise NotImplementedError
        
    @property
    @abstractmethod
    def rotation(self) -> bool:
        """ 
            True = Vertical fringes, False = Horizontal Fringes

            TODO: Maybe add support for any rotation of fringes - HARD!
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def phase(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def display(self):
        raise NotImplementedError

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

def fringe_project(camera: Camera, projector: Projector, sf, phases) -> np.ndarray:
    projector.frequency = sf
    N = len(phases)

    imgs = np.empty(shape=(N, *camera.shape))

    for i in range(N):
        projector.phase = phases[i]
        projector.display()
        imgs[i] = camera.capture().raw_data

    return imgs

# class MotorStage(Protocol):
#     @property
#     def min_height(self) -> float:
#         pass

#     @property
#     def max_height(self) -> float:
#         pass
    
#     def move_to(self, value):
#         raise NotImplementedError