import numpy as np
import cv2

from abc import ABC, abstractmethod

from .vision import VisionConfig
from .. import image, utils

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

        self.m_Debug = False

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

    @property
    def debug(self):
        return self.m_Debug
    
    @debug.setter
    def debug(self, value):
        self.m_Debug = value

    @abstractmethod
    def Characterise(self) -> bool:
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

    def PhaseMatch(self, cameraCoords, vertPhasemap, horiPhasemap, vNumStripes, hNumStripes) -> np.ndarray:
        N = cameraCoords.shape[0]

        h, w = self.config.resolution

        projCoords = np.empty((N, 2), dtype=np.float32)
    
        # This algorithm is only suited for CPU completion
        # Could implement tiling of corners for GPU accel, but pointless for now...
        vertPhasemap = utils.AlwaysNumpy(vertPhasemap)
        horiPhasemap = utils.AlwaysNumpy(horiPhasemap)

        for i, (x, y) in enumerate(cameraCoords.astype(int)):
            projCoords[i, 0] = (vertPhasemap[int(y), int(x)] * w) / (2.0 * np.pi * vNumStripes)
            projCoords[i, 1] = (horiPhasemap[int(y), int(x)] * h) / (2.0 * np.pi * hNumStripes)

        if self.debug:
            debugImg = np.zeros((h, w, 3), dtype=np.float32)

            for j, (x2, y2) in enumerate(projCoords):
                debugImg = cv2.putText(debugImg, str(j), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            image.Show(debugImg, name=f"Image: phase matched POIs")

        return projCoords

    def BilinearPhaseMatch(self, cameraCoords, vertPhasemap, horiPhasemap, vNumStripes, hNumStripes) -> np.ndarray:
        N = cameraCoords.shape[0]

        h, w = self.config.resolution

        projCoords = np.empty((N, 2), dtype=np.float32)
    
        # This algorithm is only suited for CPU completion
        # Could implement tiling of corners for GPU accel, but pointless for now...
        vertPhasemap = utils.AlwaysNumpy(vertPhasemap)
        horiPhasemap = utils.AlwaysNumpy(horiPhasemap)
        
        for i in range(N):
            x, y = cameraCoords[i]

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

            projCoords[i, 0] = (vertPhase * w) / (2.0 * np.pi * vNumStripes)

            horiPhase1 = horiPhasemap[yInt, xInt]
            horiPhase2 = horiPhasemap[yInt, xInt + 1]
            horiPhase3 = horiPhasemap[yInt + 1, xInt]
            horiPhase4 = horiPhasemap[yInt + 1, xInt + 1]

            horiPhase = (1 - yFrac) * ((1 - xFrac) * horiPhase1 + xFrac * horiPhase2) + \
                yFrac * ((1 - xFrac) * horiPhase3 + xFrac * horiPhase4)

            projCoords[i, 1] = (horiPhase * h) / (2.0 * np.pi * hNumStripes)

        if self.debug:
            debugImg = np.zeros((h, w, 3), dtype=np.float32)

            for j, (x2, y2) in enumerate(projCoords):
                debugImg = cv2.putText(debugImg, str(j), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            image.Show(debugImg, name=f"Image: phase matched POIs")

        return projCoords

    def Characterise(self, worldCoords, cameraCoords, phasemaps, vNumStripes, hNumStripes) -> bool:
        h, w = self.config.resolution

        projCoords = np.empty_like(cameraCoords)

        # Loop through each set of calibration board corner points
        for i in range(len(worldCoords)):
            projCoords[i] = self.PhaseMatch(cameraCoords[i], phasemaps[2*i], phasemaps[2*i+1], vNumStripes, hNumStripes)

        # flags = cv2.CALIB_FIX_K3
        # flags += cv2.CALIB_USE_INTRINSIC_GUESS

        # kGuess = np.array([
        #     [1110 / 912 * w, 0.00000000e+00, w / 2],
        #     [0.00000000e+00, 1110 / 570 * h, h - 1],
        #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        # ])

        # reprojErr, K, D, R, T = cv2.calibrateCamera(
        #     worldCoords, projCoords, (w, h), kGuess, None, flags=flags
        # )

        # Don't use an intrinsic guess
        reprojErr, K, D, R, T = cv2.calibrateCamera(worldCoords, projCoords, (w, h), None, None)

        projRotation = cv2.Rodrigues(R[0])[0]
        projTranslation = T[0].squeeze()

        M0 = utils.TransMat(projRotation, projTranslation)

        boardDeltaTfs = [np.eye(4, 4)]
        for i in range(1, len(worldCoords)):
            Mi = utils.TransMat(cv2.Rodrigues(R[i])[0], T[i].squeeze())
            boardDeltaTfs.append(np.linalg.inv(Mi) @ M0)

        boardDeltaTfs = np.asarray(boardDeltaTfs)

        # Calculate projector intrinsic matrix

        self.visionConfig = VisionConfig(
            rotation=projRotation, translation=projTranslation,
            intrinsicMat=K, distortMat=D, reprojErr=reprojErr,
            targetResolution=(h, w), posePOICoords=projCoords,
            boardPoses=boardDeltaTfs
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