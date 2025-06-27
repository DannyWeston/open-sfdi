import numpy as np
import cv2

from abc import ABC, abstractmethod

class VisionConfig:
    def __init__(self, rotation, translation, intrinsicMat, distortMat, reprojErr, targetResolution, posePOICoords):
        self.m_Rotation = rotation
        self.m_Translation = translation

        self.m_IntrinsicMat = intrinsicMat
        self.m_DistortMat = distortMat

        self.m_ReprojErr = reprojErr

        self.m_TargetResolution = targetResolution

        self.m_POICoords = posePOICoords

    @property
    def rotation(self) -> np.ndarray:
        return self.m_Rotation

    @property
    def translation(self) -> np.ndarray:
        return self.m_Translation
    
    @property
    def intrinsicMat(self) -> np.ndarray:
        return self.m_IntrinsicMat

    @property
    def distortMat(self) -> np.ndarray:
        return self.m_DistortMat

    @property
    def reprojErr(self) -> float:
        return self.m_ReprojErr

    @property
    def targetResolution(self) -> tuple[int, int]:
        return self.m_TargetResolution

    @property
    def posePOICoords(self) -> np.ndarray:
        return self.m_POICoords

    @property
    def extrinsicMat(self) -> np.ndarray:
        return np.concatenate([self.rotation, self.translation], axis=1)
    
    @property
    def projectionMat(self) -> np.ndarray:
        return np.dot(self.intrinsicMat, self.extrinsicMat)

class IVisionCharacterised(ABC):
    @property
    @abstractmethod
    def visionConfig(self) -> VisionConfig:
        raise NotImplementedError
    
    @visionConfig.setter
    @abstractmethod
    def visionConfig(self, value: VisionConfig):
        raise NotImplementedError


class IntensityConfig:
    def __init__(self, minIntensity, maxIntensity, coeffs, sampleCount):
        self.m_MinIntensity = minIntensity
        self.m_MaxIntensity = maxIntensity

        self.m_Coeffs = coeffs

        self.m_SampleCount = sampleCount

    @property
    def minIntensity(self) -> float:
        return self.m_MinIntensity
    
    @property
    def maxIntensity(self) -> float:
        return self.m_MaxIntensity
    
    @property
    def coeffs(self) -> np.ndarray:
        return self.m_Coeffs
    
    @property
    def sampleCount(self) -> int:
        return self.m_SampleCount
    
class IIntensityCharacterised(ABC):
    @property
    @abstractmethod
    def intensityConfig(self) -> IntensityConfig:
        raise NotImplementedError
    
def RefineDevices(device1: IVisionCharacterised, device2: IVisionCharacterised, worldCoords):
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    vc1 = device1.visionConfig
    vc2 = device2.visionConfig

    reprojErr, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        worldCoords, vc1.posePOICoords, vc2.posePOICoords,
        vc1.intrinsicMat, vc1.distortMat, vc2.intrinsicMat, vc2.distortMat,
        vc1.targetResolution, flags=flags)
    
    newVC1 = VisionConfig(vc1.rotation, vc1.translation, 
        K1, D1, vc1.reprojErr, vc1.targetResolution, vc1.posePOICoords)
    
    newVC2 = VisionConfig(np.dot(R, vc1.rotation), np.dot(R, vc1.translation) + T,
        K2, D2, vc2.reprojErr, vc2.targetResolution, vc2.posePOICoords)

    # Update vision configs of devices
    device1.visionConfig = newVC1
    device2.visionConfig = newVC2

    return reprojErr