import numpy as np
import cv2

from abc import ABC, abstractmethod

from .vision import VisionConfig, IVisionCharacterised, IIntensityCharacterised
from .. import image

class CameraConfig:
    def __init__(self, resolution, channels):
        self.m_Resolution = resolution
        self.m_Channels = channels

    @property
    def channels(self) -> int:
        return self.m_Channels

    @property
    def resolution(self) -> tuple[int, int]:
        return self.m_Resolution 

    @property
    def shape(self) -> tuple:
        if self.channels == 1:
            return self.resolution

        return (*self.resolution, self.channels)

class CalibratedCameraConfig(CameraConfig, IVisionCharacterised):
    def __init__(self, resolution, channels, visionConfig: VisionConfig):
        super().__init__(resolution, channels)

        self.m_VisionConfig = visionConfig

    @property
    def visionConfig(self) -> VisionConfig:
        return self.m_VisionConfig
       
    @visionConfig.setter
    def visionConfig(self, value: VisionConfig):
        # TODO: Maybe add callback?
        self.m_VisionConfig = value

class Camera(ABC):
    @abstractmethod
    def __init__(self, config: CameraConfig):
        self.m_UndistortOnCapture = True
        self.m_Config: CameraConfig = config

    @property
    def config(self):
        return self.m_Config
    
    @config.setter
    def config(self, value: CameraConfig):
        # TODO: Maybe add callback?
        self.m_Config = value

    @property
    def undistortOnCapture(self) -> bool:
        return self.m_UndistortOnCapture
    
    @undistortOnCapture.setter
    def undistortOnCapture(self, value):
        self.m_UndistortOnCapture = value

    def Calibrate(self, worldCoords, pixelCoords):
        h, w = self.config.resolution

        flags = cv2.CALIB_FIX_K3 + cv2.CALIB_USE_INTRINSIC_GUESS

        # Optical centre in the middle, focal length in pixels equal to resolution
        kGuess = np.array([
            [w,     0.0,    w / 2],
            [0.0,   h,      h / 2],
            [0.0,   0.0,    1.0]
        ])

        reprojErr, K, D, R, T = cv2.calibrateCamera(worldCoords, pixelCoords, (w, h), kGuess, None, flags=flags)

        visionConfig = VisionConfig(
            rotation=cv2.Rodrigues(R[0])[0], translation=T[0], 
            intrinsicMat=K, distortMat=D, reprojErr=reprojErr,
            targetResolution=(h, w), posePOICoords=pixelCoords
        )

        # errors = []
        # for i in range(len(world_xyz)):
        #     imgpoints_reproj, _ = cv2.projectPoints(
        #         world_xyz[i], R[i], t[i], self.intrinsic_mat, self.__dist_mat
        #     )
        #     error = cv2.norm(corner_pixels[i], imgpoints_reproj.reshape(-1, 2), cv2.NORM_L2) / len(imgpoints_reproj)
        #     errors.append(error)

        # # Convert to NumPy array for plotting
        # errors = np.array(errors)

        # TODO: Keep all poses

        return CalibratedCameraConfig(self.config.resolution, self.config.channels, visionConfig)

    def Undistort(self, img):
        # Can't undistort if camera is not calibrated
        if not isinstance(self.config, CalibratedCameraConfig):
            return img

        # Optional: New camera matrix (use original K_cam to preserve resolution)
        return image.Undistort(img, self.config.visionConfig)

    @abstractmethod
    def Capture(self) -> image.Image:
        """ Capture an image NOTE: should return pixels with float32 type """
        raise NotImplementedError

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

    def Capture(self):
        # Capture an image
        if self.__raw_camera is None: 
            self.__raw_camera = cv2.VideoCapture(self.device)
            self.__set_active_res(self.resolution)
        
        ret, raw_image = self.__raw_camera.read()

        if not ret:
            # Couldn't capture image, throw exception
            raise Exception("Could not capture an image with the CV2Camera")
        
        # Must convert to float32, spec!
        raw_image = image.ToF32(raw_image)

        # Convert to grey if needed
        if self.channels == 1:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

        return image.Image(self.Undistort(raw_image))

    def show_feed(self):
        cv2.namedWindow("Camera feed")
        
        while img := self.Capture():
            cv2.imshow("Camera feed", img.raw_data)

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

        cv2.destroyWindow("Camera feed")

    def __del__(self):
        if self.__raw_camera:
            if self.__raw_camera.isOpened():
                self.__raw_camera.release()

class FileCamera(Camera):
    def __init__(self, config: CameraConfig, imgs=None):
        super().__init__(config)

        self.m_Images: list[image.FileImage] = imgs

    @property
    def images(self):
        return self.m_Images
    
    @images.setter
    def images(self, value):
        self.m_Images = value

    def Capture(self) -> image.Image:
        try:
            # This will invoke the loading of the data from the FileImage (disk) 
            img_data = self.m_Images.pop(0).raw_data

            # Apply undistortion if needed
            if self.undistortOnCapture:
                img_data = self.Undistort(img_data)

            return image.Image(img_data)
        except IndexError:
            return None