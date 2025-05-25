import numpy as np
import cv2

from abc import ABC, abstractmethod
from . import image


# Camera stuff

class CameraRegistry:
    _registry = {}

    @classmethod
    def register(cls, clazz):
        cls._registry[clazz.__name__] = clazz
        return clazz

    @classmethod
    def get_class(cls, name):
        return cls._registry[name]
    
    @classmethod
    def is_registered(cls, clazz):
        return clazz.__name__ in cls._registry.keys()

class Camera(ABC):
    @abstractmethod
    def __init__(self, resolution=(720, 1280), channels=1, apply_undistort=True):
        self.__proj_mat = None
        self.__dist_mat = None
        self.reproj_error = None

        self.__apply_undistort = apply_undistort

        self.intrinsic_mat: np.ndarray = None

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

    @property
    def apply_undistort(self) -> bool:
        return self.__apply_undistort
    
    @apply_undistort.setter
    def apply_undistort(self, value):
        self.__apply_undistort = value

    def is_calibrated(self) -> bool:
        return not (self.proj_mat is None)

    def calibrate(self, world_xyz, corner_pixels):
        h, w = self.resolution

        self.reproj_error, self.intrinsic_mat, self.__dist_mat, R, t = cv2.calibrateCamera(world_xyz, 
            corner_pixels, (w, h), None, None)

        # errors = []
        # for i in range(len(world_xyz)):
        #     imgpoints_reproj, _ = cv2.projectPoints(
        #         world_xyz[i], R[i], t[i], self.intrinsic_mat, self.__dist_mat
        #     )
        #     error = cv2.norm(corner_pixels[i], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
        #     errors.append(error)

        # # Convert to NumPy array for plotting
        # errors = np.array(errors)

        R, _  = cv2.Rodrigues(R[0])
        t = t[0]

        # Calculate camera projection matrix (use first entry, could use any)
        self.proj_mat = self.intrinsic_mat @ np.hstack((R, t.reshape(3, 1)))

        return self.reproj_error

    def undistort(self, img):
        # Can't undistort if camera is not calibrated
        if not self.is_calibrated():
            return img

        # Optional: New camera matrix (use original K_cam to preserve resolution)

        return image.undistort_img(img, self.intrinsic_mat, self.dist_mat)

    @abstractmethod
    def capture(self) -> image.Image:
        """ Capture an image NOTE: should return pixels with float32 type """
        raise NotImplementedError

@CameraRegistry.register
class CV2Camera(Camera):
    def __init__(self, device, resolution=(720, 1280), channels=3):
        super().__init__(resolution, channels)

        self.__device = device
        self.__raw_camera = None

    @property
    def device(self):
        return self.__device
    
    @device.setter
    def device(self, value):
        if value != self.__device:
            self.__raw_camera = None # Need to reopen the feed

        self.__device = value

    @property
    def resolution(self) -> tuple[int, int]:
        return super().resolution
    
    @resolution.setter
    def resolution(self, value):
        super().resolution = value

        if self.__raw_camera:
            self.__set_active_res(value)

    def __set_active_res(self, value):
        self.__raw_camera.set(cv2.CAP_PROP_FRAME_WIDTH, value[1])
        self.__raw_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, value[0])

    def capture(self):
        # Capture an image
        if self.__raw_camera is None: 
            self.__raw_camera = cv2.VideoCapture(self.device)
            self.__set_active_res(self.resolution)
        
        ret, raw_image = self.__raw_camera.read()

        if not ret:
            # Couldn't capture image, throw exception
            raise Exception("Could not capture an image with the CV2Camera")
        
        # Must convert to float32, spec!
        raw_image = image.to_f32(raw_image)

        # Convert to grey if needed
        if self.channels == 1:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

        return image.Image(self.undistort(raw_image))

    def show_feed(self):
        cv2.namedWindow("Camera feed")
        
        while img := self.capture():
            cv2.imshow("Camera feed", img.raw_data)

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

        cv2.destroyWindow("Camera feed")

    def __del__(self):
        if self.__raw_camera:
            if self.__raw_camera.isOpened():
                self.__raw_camera.release()

@CameraRegistry.register
class FileCamera(Camera):
    def __init__(self, imgs=None, resolution=(720, 1280), channels=1):
        super().__init__(resolution, channels)

        self.__imgs: list[image.FileImage] = imgs

    @property
    def imgs(self):
        return self.__imgs
    
    @imgs.setter
    def imgs(self, value):
        self.__imgs = value

    def capture(self) -> image.Image:
        try:
            # This will invoke the loading of the data from the FileImage (disk) 
            img_data = self.__imgs.pop(0).raw_data

            # Apply undistortion if needed
            if self.apply_undistort:
                img_data = self.undistort(img_data)

            return image.Image(img_data)
        except IndexError:
            return None

# Projector stuff

class ProjectorRegistry:
    _registry = {}

    @classmethod
    def register(cls, clazz):
        cls._registry[clazz.__name__] = clazz
        return clazz

    @classmethod
    def get_class(cls, name):
        return cls._registry[name]
    
    @classmethod
    def is_registered(cls, clazz):
        return clazz.__name__ in cls._registry.keys()

class Projector(ABC):
    @abstractmethod
    def __init__(self, resolution=(720, 1280), channels=1):
        self.__proj_mat = None
        self.__dist_mat = None
        self.intrinsic_mat = None

        self.reproj_error = None

        self.__resolution = resolution
        self.__channels = channels
        self.__rotation = 0.0
        self.__phase = 0.0
        self.__spatial_freq = 8.0

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
    def frequency(self) -> float:
        return self.__spatial_freq
    
    @frequency.setter
    def frequency(self, value):
        if value < 0.0:
            return
        
        self.__spatial_freq = value

    @property
    def phase(self) -> float:
        return self.__phase
    
    @phase.setter
    def phase(self, value):
        if value < 0.0:
            return
        
        self.__phase = value

    @property
    def rotation(self) -> bool:
        return self.__rotation
    
    @rotation.setter
    def rotation(self, value):
        self.__rotation = value

    @property
    def resolution(self) -> tuple[int, int]:
        return self.__resolution
    
    @resolution.setter
    def resolution(self, value):
        self.__resolution = value

    @property
    def channels(self):
        return self.__channels
    
    @channels.setter
    def channels(self, value):
        self.__channels = value

    def calibrate(self, world_xyz, corner_subpixels, phasemaps, num_stripes):
        # DEPRECATED: Convert camera corner pixels to ints for indexing
        # Use bilinear interp as better accuracy
        corner_pixels = np.round(corner_subpixels).astype(int)

        # For the detected checkerboard corners, these are subpixel coordinates
        # Use bilinear interp to convert to subpixel coordinates for the phase values
        # np.nanmean() is used because the phasemap may contain nans for any filtered values

        h, w = self.resolution

        proj_coords = np.empty_like(corner_subpixels, dtype=np.float32)

        # Loop through each set of calibration board corner points
        for cb_i in range(len(world_xyz)):            
            corner_xs = corner_pixels[cb_i, :, 0] # Vertical corner pixels
            corner_ys = corner_pixels[cb_i, :, 1]

            phi_v = phasemaps[2*cb_i]    # Vertical fringesphasemaps
            phi_h = phasemaps[2*cb_i+1]  # Horizontal fringe phasemaps

            # Convert camera coordinates to projector coordinates
            # "act as the projector's eye"
            x_p = (phi_v[corner_ys, corner_xs] * w) / (2.0 * np.pi * num_stripes)
            y_p = (phi_h[corner_ys, corner_xs] * h) / (2.0 * np.pi * num_stripes)

            proj_coords[cb_i] = np.dstack((x_p, y_p))

        self.reproj_error, self.intrinsic_mat, self.__dist_mat, R, t = cv2.calibrateCamera(world_xyz, proj_coords, (w, h), None, None)

        R, _  = cv2.Rodrigues(R[0])
        t = t[0]

        # Calculate camera projection matrix (use first entry, could use any)
        self.proj_mat = self.intrinsic_mat @ np.hstack((R, t.reshape(3, 1)))

        return self.reproj_error

    def is_calibrated(self) -> bool:
        return not (self.proj_mat is None)

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

class DisplayProjector(Projector):
    def __init__(self, resolution=(720, 1280), channels=1):
        super().__init__(resolution=resolution, channels=channels)

    def display(self):
        # Do something to display stuff on a screen
        pass

# TODO: Add FileProjector, could be useful in the future

def fringe_project(camera: Camera, projector: Projector, sf, phases) -> np.ndarray:
    projector.frequency = sf
    N = len(phases)

    imgs = np.empty(shape=(N, *camera.shape))

    for i in range(N):
        projector.phase = phases[i]
        projector.display()
        imgs[i] = camera.capture().raw_data

    return imgs