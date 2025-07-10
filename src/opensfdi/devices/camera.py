import numpy as np
import cv2

from abc import ABC, abstractmethod

from .vision import VisionConfig
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

class Camera(ABC):
    @abstractmethod
    def __init__(self, config: CameraConfig, visionConfig:VisionConfig=None):
        self.m_ShouldUndistort = True

        self.m_Config: CameraConfig = config
        self.m_VisionConfig = visionConfig

    @property
    def visionConfig(self) -> VisionConfig:
        return self.m_VisionConfig
    
    @visionConfig.setter
    def visionConfig(self, visionConfig: VisionConfig):
        self.m_VisionConfig = visionConfig

    @property
    def config(self):
        return self.m_Config
    
    @config.setter
    def config(self, value: CameraConfig):
        # TODO: Maybe add callback?
        self.m_Config = value

    @property
    def shouldUndistort(self) -> bool:
        return self.m_ShouldUndistort
    
    @shouldUndistort.setter
    def shouldUndistort(self, value):
        self.m_ShouldUndistort = value

    def Calibrate(self, worldCoords, pixelCoords):
        h, w = self.config.resolution

        flags = 0
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        # Optical centre in the middle, focal length in pixels equal to resolution
        # 1,838.2978723404255319148936170213
        # 1,418.9781021897810218978102189781

        sensorWidth = 3.76 # mm
        focalX = (3.6 / sensorWidth) * w # mm
        focalY = focalX # TODO: Incorporate pixelSize for camera

        kGuess = np.array([
            [focalX,        0.0,    w/2],
            [0.0,           focalY, h/2],
            [0.0,           0.0,    1.0]
        ])

        reprojErr, K, D, R, T = cv2.calibrateCamera(worldCoords, pixelCoords, (w, h), kGuess, None, flags=flags)

        self.visionConfig = VisionConfig(
            rotation=cv2.Rodrigues(R[0])[0], translation=T[0], 
            intrinsicMat=K, distortMat=D, reprojErr=reprojErr,
            targetResolution=(h, w), posePOICoords=pixelCoords
        )

        # errors = []
        # for i in range(len(worldCoords)):
        #     points, _ = cv2.projectPoints(
        #         worldCoords[i], R[i], T[i], K, D
        #     )
        #     error = cv2.norm(pixelCoords[i], points.reshape(-1, 2), cv2.NORM_L2) / len(points)
        #     errors.append(error)

        # print(np.array(errors))

        # TODO: Keep all poses
        return True

    def Undistort(self, img):
        # Can't undistort image without characterisation
        # TODO: Add warning
        if self.visionConfig is None: return img
        
        # Optional: New camera matrix (use original K_cam to preserve resolution)
        return image.Undistort(img, self.visionConfig)

    @abstractmethod
    def Capture(self) -> image.Image:
        """ Capture an image NOTE: should return pixels with float32 type """
        raise NotImplementedError

class CV2Camera(Camera):
    def __init__(self, config: CameraConfig, device, visionConfig: VisionConfig=None):
        super().__init__(config, visionConfig=visionConfig)

        self.m_Device = device
        self.m_CameraHandle = None

    @property
    def device(self):
        return self.m_Device
    
    @device.setter
    def device(self, value):
        if value != self.m_Device:
            self.m_CameraHandle = None # Need to reopen the feed

        self.m_Device = value

    def SetActiveResolution(self, value):
        self.m_CameraHandle.set(cv2.CAP_PROP_FRAME_WIDTH, value[1])
        self.m_CameraHandle.set(cv2.CAP_PROP_FRAME_HEIGHT, value[0])

    def Capture(self):
        # Capture an image
        if self.m_CameraHandle is None: 
            self.m_CameraHandle = cv2.VideoCapture(self.device)
            self.SetActiveResolution(self.config.resolution)
        
        ret, rawImage = self.m_CameraHandle.read()

        if not ret:
            # Couldn't capture image, throw exception
            raise Exception("Could not capture an image with the CV2Camera")
        
        # Must convert to float32, spec!
        rawImage = image.ToF32(rawImage)

        # Convert to grey if needed
        if self.config.channels == 1:
            rawImage = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)

        if self.shouldUndistort:
            rawImage = self.Undistort(rawImage)

        return image.Image(rawImage)

    def ShowFeed(self):
        cv2.namedWindow("Camera feed")
        
        while img := self.Capture():
            cv2.imshow("Camera feed", img.rawData)

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

        cv2.destroyWindow("Camera feed")

    def __del__(self):
        if self.m_CameraHandle:
            if self.m_CameraHandle.isOpened():
                self.m_CameraHandle.release()

class FileCamera(Camera):
    def __init__(self, config: CameraConfig, visionConfig: VisionConfig=None, images:list[image.FileImage]=None):
        super().__init__(config, visionConfig=visionConfig)

        self.m_Images = images

    @property
    def images(self) -> list[image.FileImage]:
        return self.m_Images
    
    @images.setter
    def images(self, value):
        self.m_Images = value

    def Capture(self) -> image.Image:
        try:
            # This will invoke the loading of the data from the FileImage (disk) 
            rawImage = self.m_Images.pop(0).rawData

            # Apply undistortion if needed
            if self.shouldUndistort:
                rawImage = self.Undistort(rawImage)

            return image.Image(rawImage)
        except IndexError:
            return None