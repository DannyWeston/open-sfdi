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
        self.__channels = channels
        self.__resolution = resolution
        self.__apply_undistort = apply_undistort

        self.calibrated = False

        self.K: np.ndarray = None
        self.R = None
        self.t = None

        self.__dist_mat = None
        self.reproj_error = None

    @property
    def dist_mat(self) -> np.ndarray:
        return self.__dist_mat

    @dist_mat.setter
    def dist_mat(self, value):
        self.__dist_mat = value

    @property
    def ext_mat(self):
        return np.concatenate([self.R, self.t], axis=1)

    @property
    def proj_mat(self) -> np.ndarray:
        return np.dot(self.K, self.ext_mat)

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

    def calibrate(self, world_xyz, corner_pixels):
        h, w = self.resolution

        flags = cv2.CALIB_FIX_K3 + cv2.CALIB_USE_INTRINSIC_GUESS

        # Optical centre in the middle, focal length in pixels equal to resolution
        K_guess = np.array([
            [w,     0.0,    w / 2],
            [0.0,   h,      h / 2],
            [0.0,   0.0,    1.0]
        ])

        self.reproj_error, self.K, self.__dist_mat, R, t = cv2.calibrateCamera(world_xyz, 
            corner_pixels, (w, h), K_guess, None, flags=flags)

        # errors = []
        # for i in range(len(world_xyz)):
        #     imgpoints_reproj, _ = cv2.projectPoints(
        #         world_xyz[i], R[i], t[i], self.intrinsic_mat, self.__dist_mat
        #     )
        #     error = cv2.norm(corner_pixels[i], imgpoints_reproj.reshape(-1, 2), cv2.NORM_L2) / len(imgpoints_reproj)
        #     errors.append(error)

        # # Convert to NumPy array for plotting
        # errors = np.array(errors)

        self.calibrated = True
        self.R, _  = cv2.Rodrigues(R[0])
        self.t = t[0]

        return self.reproj_error

    def undistort(self, img):
        # Can't undistort if camera is not calibrated
        if not self.calibrated:
            return img

        # Optional: New camera matrix (use original K_cam to preserve resolution)

        return image.undistort_img(img, self.K, self.dist_mat)

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
    def __init__(self, resolution=(720, 1280), channels=1, throw_ratio=1.0, pixel_size=1.0):
        self.__resolution = resolution
        self.__channels = channels
        self.__rotation = 0.0
        self.__phase = 0.0
        self.__spatial_freq = 8.0

        self.__throw_ratio = throw_ratio    # Projector throw ratio 
        self.__pixel_size = pixel_size      # w / h pixels

        self.proj_coords = None

        self.calibrated = False

        # Intrinsic / rotation / translation
        self.K: np.ndarray = None
        self.R: np.ndarray = None
        self.t: np.ndarray = None

        self.__dist_mat = None
        self.reproj_error = None

    @property
    def dist_mat(self) -> np.ndarray:
        return self.__dist_mat
    
    @dist_mat.setter
    def dist_mat(self, value):
        self.__dist_mat = value

    @property
    def throw_ratio(self):
        return self.__throw_ratio
    
    @throw_ratio.setter
    def throw_ratio(self, value):
        self.__throw_ratio = value
    
    @property
    def pixel_size(self):
        return self.__pixel_size

    @pixel_size.setter
    def pixel_size(self, value):
        self.__pixel_size = value

    @property
    def extr_mat(self):
        return np.concatenate([self.R, self.t], axis=1)

    @property
    def proj_mat(self) -> np.ndarray:
        return np.dot(self.K, self.extr_mat)

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

    def _phase_match(self, corners, phi_v, phi_h, num_stripes):
        N = corners.shape[0]
        proPoints = np.empty((N, 2), dtype=np.float32)
        
        for i in range(N):
            x, y = corners[i]

            pRowUp = int(y)
            pRowLow = pRowUp + 1
            pColLeft = int(x)
            pColRight = pColLeft + 1
            rowRatio = y - pRowUp
            colRatio = x - pColLeft

            phaseVA = phi_v[pRowUp, pColLeft]
            phaseVB = phi_v[pRowUp, pColRight]
            phaseVC = phi_v[pRowLow, pColLeft]
            phaseVD = phi_v[pRowLow, pColRight]
            phaseVP = (1 - rowRatio) * ((1 - colRatio) * phaseVA + colRatio * phaseVB) + rowRatio * ((1 - colRatio) * phaseVC + colRatio * phaseVD)

            proCol = phaseVP / np.pi / (2.0 * num_stripes)
            proPoints[i, 0] = proCol

            phaseHA = phi_h[pRowUp, pColLeft]
            phaseHB = phi_h[pRowUp, pColRight]
            phaseHC = phi_h[pRowLow, pColLeft]
            phaseHD = phi_h[pRowLow, pColRight]
            phaseHP = (1 - rowRatio) * ((1 - colRatio) * phaseHA + colRatio * phaseHB) + rowRatio * ((1 - colRatio) * phaseHC + colRatio * phaseHD)

            proRow = phaseHP / np.pi / (2.0 * num_stripes)
            proPoints[i, 1] = proRow

        return proPoints

    def calibrate(self, world_xyz, corner_subpixels, phasemaps, num_stripes):
        h, w = self.resolution

        self.proj_coords = np.empty_like(corner_subpixels, dtype=np.float32)

        # Loop through each set of calibration board corner points
        for cb_i in range(len(world_xyz)):
            self.proj_coords[cb_i] = self._phase_match(corner_subpixels[cb_i], 
                phasemaps[2*cb_i], phasemaps[2*cb_i+1], num_stripes)

        # Optical centre in the middle, focal length in pixels equal to resolution
        K_guess = np.array([
            [self.throw_ratio * w,  0.0,                                    w / 2],
            [0.0,                   self.throw_ratio*self.pixel_size*h,     h / 2],
            [0.0,                   0.0,                                    1.0]
        ])

        flags = 0
        # flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        self.reproj_error, self.K, self.__dist_mat, R, t = cv2.calibrateCamera(world_xyz, self.proj_coords, (w, h), 
            K_guess, None, flags=flags)
        
        # Calculate projector intrinsic matrix
        self.R, _  = cv2.Rodrigues(R[0])
        self.t = t[0]

        self.calibrated = True

        return self.reproj_error

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

@ProjectorRegistry.register
class FakeProjector(Projector):
    def __init__(self, resolution=(1080, 1920), channels=1, throw_ratio=1.0, pixel_size=1.0):
        super().__init__(resolution=resolution, channels=channels, throw_ratio=throw_ratio, pixel_size=pixel_size)

    def display(self):
        # Do nothing
        pass
# TODO: Add FileProjector, could be useful in the future


# CalibrationBoard

class CalibrationBoard(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def find_pois(self, img: np.ndarray):
        raise NotImplementedError
    
    def get_poi_coords(self):
        raise NotImplementedError

class Checkerboard(CalibrationBoard):
    def __init__(self, square_width=0.018, poi_count=(10, 7)):
        self.__poi_count = poi_count
        self.__square_width = square_width

        # Multiply by square width
        w, h = self.__poi_count
        self.__cb_corners = np.zeros((w * h, 3), np.float32)
        self.__cb_corners[:, :2] = np.mgrid[:w, :h].T.reshape(-1, 2) * self.__square_width

    def find_pois(self, img: np.ndarray):
        # Change image to int if not already
        uint_img = image.to_int8(img)

        # flags = cv2.CALIB_CB_EXHAUSTIVE
        # result, corners = cv2.findChessboardCornersSB(uint_img, cb_size, flags=flags)
        # if not result: return None

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_EXHAUSTIVE
        result, corners = cv2.findChessboardCorners(uint_img, self.__poi_count, flags=flags)

        if not result: return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(uint_img, corners, (15, 15), (-1, -1), criteria)

        return corners.squeeze()

    def get_poi_coords(self):
        return self.__cb_corners.copy()

class CircleBoard(CalibrationBoard):
    
    def __init__(self, circle_diameter=1.0, circle_spacing=1.0, poi_count=(10, 7), inverted=True, staggered=True, area_hint=None):
        self.__poi_count = poi_count
        self.__diameter = circle_diameter
        self.__spacing = circle_spacing

        self.__staggered = staggered

        # Setup blob detector
        self.__detector_params = cv2.SimpleBlobDetector_Params()

        if area_hint is not None:
            # 480x270   :   (100, 400)
            # 640x360   :   (200, 1000)
            # 960x540   :   (500, 2000)
            # 1440x810  :   (1250, 5000)
            # 1920x1080 :   (2500, 13000)
            # 2560x1440 :   (5000, 20000)
            # 3840x2160 :   (7000, 28000)

            self.__detector_params.filterByArea = True
            self.__detector_params.minArea = area_hint[0]
            self.__detector_params.maxArea = area_hint[1]

        self.__detector_params.blobColor = 255 if inverted else 0

        # Multiply by square width
        w, h = self.__poi_count

        # Setup checkerboard world coordinates
        if self.__staggered:
            self.__circle_centres = np.zeros((w * h, 3), dtype=np.float32)
            for i in range(h):
                    for j in range(w):
                        self.__circle_centres[i * w + j, :2] = [(2 * j + i % 2) * self.__spacing / 2, i * self.__spacing / 2]
        else:
            self.__circle_centres = np.zeros((w * h, 3), np.float32)
            self.__circle_centres[:, :2] = np.mgrid[:w, :h].T.reshape(-1, 2) * self.__spacing

    def find_pois(self, img: np.ndarray):
        # Mask the image
        img = image.threshold_mask(img, threshold=0.1)

        # Convert image to uint datatype
        uint_img = image.to_int8(img)

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(self.__detector_params)

        flags = (cv2.CALIB_CB_ASYMMETRIC_GRID if self.__staggered else cv2.CALIB_CB_SYMMETRIC_GRID)

        result, corners = cv2.findCirclesGrid(uint_img, self.__poi_count, blobDetector=detector, flags=flags)

        if not result: return None

        # DEBUG: Show the corners on the image
        # corners_img = cv2.drawChessboardCorners(uint_img, self.__poi_count, corners, result)
        # image.show_image(corners_img)

        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # corners = cv2.cornerSubPix(uint_img, corners, (15, 15), (-1, -1), criteria)

        return corners.squeeze()

    def get_poi_coords(self):
        return self.__circle_centres.copy()

def fringe_project(camera: Camera, projector: Projector, sf, phases) -> np.ndarray:
    projector.frequency = sf
    N = len(phases)

    imgs = np.empty(shape=(N, *camera.shape))

    for i in range(N):
        projector.phase = phases[i]
        projector.display()
        imgs[i] = camera.capture().raw_data
        # imgs[i] = image.add_gaussian(imgs[i], sigma=0.1)

    return imgs