import numpy as np
import cv2

from abc import ABC, abstractmethod

from .vision import Characterisation, ICharacterisable
from .. import image, utils


class Camera(ICharacterisable):
    @abstractmethod
    def __init__(self, resolution, channels, refreshRate, character:Characterisation=None):
        self.m_ShouldUndistort = True

        self.m_Characterisation:Characterisation = character

        self.m_Debug = False

        self.m_Channels = channels
        self.m_Resolution = resolution
        self.m_RefreshRate = refreshRate

    @property
    def channels(self) -> int:
        return self.m_Channels

    @property
    def resolution(self) -> tuple[int, int]:
        return self.m_Resolution 
    
    @property
    def refreshRate(self) -> float:
        return self.m_RefreshRate

    @property
    def shape(self) -> tuple:
        if self.channels == 1:
            return self.resolution

        return (*self.resolution, self.channels)

    @property
    def characterisation(self) -> Characterisation:
        return self.m_Characterisation
    
    def Characterise(self, worldCoords, poiCoords):
        extraFlags = cv2.CALIB_FIX_PRINCIPAL_POINT + \
            cv2.CALIB_FIX_ASPECT_RATIO

        # Could return some information about the characterisation
        return self.characterisation.Calculate(self.resolution, worldCoords, poiCoords, 
                                        extraFlags=None)
    
    @property
    def debug(self):
        return self.m_Debug

    @debug.setter
    def debug(self, value):
        self.m_Debug = value

    @characterisation.setter
    def characterisation(self, visionConfig: Characterisation):
        self.m_Characterisation = visionConfig

    @property
    def shouldUndistort(self) -> bool:
        return self.m_ShouldUndistort
    
    @shouldUndistort.setter
    def shouldUndistort(self, value):
        self.m_ShouldUndistort = value

    

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
        return image.Undistort(img, self.characterisation)

    @abstractmethod
    def Capture(self) -> image.Image:
        """ Capture an image NOTE: should return pixels with float32 type """
        raise NotImplementedError

class CV2Camera(Camera):
    def __init__(self,  resolution, channels, refreshRate, device, character: Characterisation=None):
        super().__init__(resolution, channels, refreshRate, character=character)

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
            self.SetActiveResolution(self.resolution)
        
        ret, rawImage = self.m_CameraHandle.read()

        if not ret:
            # Couldn't capture image, throw exception
            raise Exception("Could not capture an image with the CV2Camera")
        
        # Must convert to float32, spec!
        rawImage = image.ToFloat(rawImage)

        # Convert to grey if needed
        if self.channels == 1:
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
    def __init__(self, resolution, channels, refreshRate, character: Characterisation=None, images:list[image.FileImage]=None):
        super().__init__(resolution, channels, refreshRate, character=character)

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