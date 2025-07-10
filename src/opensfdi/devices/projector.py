import numpy as np
import cv2

from abc import ABC, abstractmethod

from .vision import VisionConfig
from ..image import Show

class ProjectorConfig:
    def __init__(self, resolution=(720, 1280), channels=1, throwRatio=1.0, pixelSize=1.0):
        self.m_Resolution = resolution  # h, w
        self.m_Channels = channels

        self.m_ThrowRatio = throwRatio
        self.m_PixelSize = pixelSize    # w / h pixels

    @property
    def resolution(self) -> tuple[int, int]:
        return self.m_Resolution
    
    @property
    def channels(self) -> int:
        return self.m_Channels
    
    @property
    def throwRatio(self) -> float:
        return self.m_ThrowRatio
    
    @property
    def pixelSize(self) -> float:
        return self.m_PixelSize
    
    @property
    def shape(self):
        if self.channels == 1:
            return self.resolution
        
        return (*self.resolution, self.channels)

class Projector(ABC):
    @abstractmethod
    def __init__(self, config: ProjectorConfig, visionConfig:VisionConfig=None):
        self.m_Config = config

        self.m_ShouldUndistort = False
        self.m_VisionConfig = visionConfig

    @property
    def config(self):
        return self.m_Config

    @config.setter
    def config(self, value: ProjectorConfig):
        # TODO: Maybe add a callback?
        self.m_Config = value

    @property
    def visionConfig(self) -> VisionConfig:
        return self.m_VisionConfig
    
    @visionConfig.setter
    def visionConfig(self, value: VisionConfig):
        # TODO: Maybe add callback?
        self.m_VisionConfig = value

    @property
    def shouldUndistort(self) -> bool:
        return self.m_ShouldUndistort
    
    @shouldUndistort.setter
    def shouldUndistort(self, value):
        self.m_ShouldUndistort = value

    @abstractmethod
    def Calibrate(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def Display(self):
        raise NotImplementedError

class FringeProjector(Projector):
    def __init__(self, config: ProjectorConfig, visionConfig: VisionConfig=None, stripeRotation=0.0, phase=0.0, numStripes=0.0):
        super().__init__(config, visionConfig=visionConfig)

        self.m_StripeRotation = stripeRotation
        self.m_Phase = phase
        self.m_NumStripes = numStripes

    @property
    def numStripes(self) -> float:
        return self.m_NumStripes
    
    @numStripes.setter
    def numStripes(self, value):
        if value < 0: return
        
        self.m_NumStripes = value

    @property
    def phase(self) -> float:
        return self.m_Phase
    
    @phase.setter
    def phase(self, value):
        self.m_Phase = value

    @property
    def stripeRotation(self) -> float:
        return self.m_StripeRotation
    
    @stripeRotation.setter
    def stripeRotation(self, value):
        self.m_StripeRotation = value

    def PhaseMatch(self, worldCoords, vertPhasemap, horiPhasemap, numStripes) -> np.ndarray:
        N = worldCoords.shape[0]

        h, w = self.config.resolution

        projCoords = np.empty((N, 2), dtype=np.float32)
        
        for i in range(N):
            x, y = worldCoords[i]

            xFrac, xInt = np.modf(x)
            yFrac, yInt = np.modf(y)
            
            xInt = int(xInt)
            yInt = int(yInt)

            vertPhase1 = vertPhasemap[yInt, xInt]
            vertPhase2 = vertPhasemap[yInt, xInt + 1]
            vertPhase3 = vertPhasemap[yInt + 1, xInt]
            vertPhase4 = vertPhasemap[yInt + 1, xInt + 1]

            vertPhase = (1 - yFrac) * ((1 - xFrac) * vertPhase1 + xFrac * vertPhase2) + \
                yFrac * ((1 - xFrac) * vertPhase3 + xFrac * vertPhase4)

            projCoords[i, 0] = (vertPhase * w) / (2.0 * np.pi * numStripes)

            horiPhase1 = horiPhasemap[yInt, xInt]
            horiPhase2 = horiPhasemap[yInt, xInt + 1]
            horiPhase3 = horiPhasemap[yInt + 1, xInt]
            horiPhase4 = horiPhasemap[yInt + 1, xInt + 1]

            horiPhase = (1 - yFrac) * ((1 - xFrac) * horiPhase1 + xFrac * horiPhase2) + \
                yFrac * ((1 - xFrac) * horiPhase3 + xFrac * horiPhase4)

            projCoords[i, 1] = (horiPhase * h) / (np.pi * 2.0 * numStripes)

        return projCoords

    def Calibrate(self, worldCoords, cameraCoords, phasemaps, numStripes) -> bool:
        h, w = self.config.resolution

        pixelCoords = np.empty_like(cameraCoords)

        # Loop through each set of calibration board corner points
        for i in range(len(worldCoords)):
            pixelCoords[i] = self.PhaseMatch(cameraCoords[i], phasemaps[2*i], phasemaps[2*i+1], numStripes)

        # for coords in pixelCoords:
        #     cornersImage = cv2.drawChessboardCorners(np.zeros(self.config.resolution, dtype=np.float32), (4, 13), coords, True) 
        #     cornersImage = cv2.circle(cornersImage, coords[0].astype(int), 10, 255, -1)
        #     Show(cornersImage)

        sensorWidth     = 9.962108389 # mm
        sensorHeight    = 5.603685969 # mm
        focalX = (3.6 / sensorWidth) * w # mm
        focalY = focalX # TODO: Incorporate pixelSize for camera

        # Optical centre in the middle, focal length in pixels equal to resolution
        K_guess = np.array([
            [focalX,    0.0,        w / 2],
            [0.0,       focalY,     h / 2],
            [0.0,       0.0,        1.0]
        ])

        flags = 0
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5

        reprojErr, K, D, R, T = cv2.calibrateCamera(worldCoords, pixelCoords, (w, h), None, None, flags=flags)
        
        # Calculate projector intrinsic matrix

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

        return True
  
    @abstractmethod
    def Display(self):
        raise NotImplementedError