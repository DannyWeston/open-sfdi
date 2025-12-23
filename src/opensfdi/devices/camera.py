import cv2

from abc import abstractmethod

from .characterisation import Characterisation, ICharacterisable, CalibrationBoard
from .. import image, utils

class Camera(ICharacterisable):
    @abstractmethod
    def __init__(self, resolution, channels, refreshRate, character:Characterisation=None):
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
    
    def Characterise(self, board: CalibrationBoard, poiCoords, extraFlags=None):
        # TODO: add extra flags
        defaultFlags = cv2.CALIB_FIX_PRINCIPAL_POINT + \
            cv2.CALIB_FIX_ASPECT_RATIO

        # Could return some information about the characterisation
        return self.characterisation.Calculate(board, poiCoords, self.resolution, defaultFlags)
    
    @property
    def debug(self):
        return self.m_Debug

    @debug.setter
    def debug(self, value):
        self.m_Debug = value

    @characterisation.setter
    def characterisation(self, visionConfig: Characterisation):
        self.m_Characterisation = visionConfig

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

    def Capture(self) -> image.Image:
        xp = utils.ProcessingContext().xp

        # Capture an image
        if self.m_CameraHandle is None: 
            self.m_CameraHandle = cv2.VideoCapture(self.device)
            self.SetActiveResolution(self.resolution)
        
        ret, rawImg = self.m_CameraHandle.read()

        if not ret:
            # Couldn't capture image, throw exception
            raise Exception("Could not capture an image with the CV2Camera")
        
        # Use float (spec of program)!
        rawImg = image.ToFloat(rawImg)

        # Undistort if can
        rawImg = self.characterisation.Undistort(rawImg)

        # Convert to grey if needed
        if self.channels == 1:
            rawImg = cv2.cvtColor(rawImg, cv2.COLOR_BGR2GRAY)

        return image.Image(xp.asarray(rawImg))

    def ShowFeed(self):
        cv2.namedWindow("Camera feed")
        
        while img := self.Capture():
            rawImg = utils.ToNumpy(img.rawData)

            # Apply undistortion if needed
            rawImg = self.characterisation.Undistort(rawImg)

            cv2.imshow("Camera feed", rawImg)

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
            return self.m_Images.pop(0)
        
        except IndexError:
            return None