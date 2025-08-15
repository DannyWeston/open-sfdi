import numpy as np
import cv2

from abc import ABC, abstractmethod

class VisionConfig:
    
    # Rotation and Translation are the transformation from the characterisation board's frame to the camera frame.
    # The first pose (Oxyz1) is used to determine the camera offset from the board along with the Rotation (R) and Translation (T):
    # I.e: Cxyz1 =  (R   T) . Oxyz1
    #               (0   1)
    #
    # The list of in-calibration order board poses are described by: 
    # Oxyz_i =  (R_i   T_i) . Oxyz1 {i -> 0 .. poses used}
    #           (0     1)

    def __init__(self, rotation, translation, intrinsicMat, distortMat, reprojErr, targetResolution, posePOICoords, boardPoses):
        self.m_Rotation = rotation
        self.m_Translation = translation

        self.m_IntrinsicMat = intrinsicMat
        self.m_DistortMat = distortMat

        self.m_ReprojErr = reprojErr

        self.m_TargetResolution = targetResolution

        self.m_POICoords = posePOICoords

        self.m_BoardPoses = boardPoses

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
    def boardPoses(self) -> np.ndarray:
        return self.m_BoardPoses

    @property
    def posePOICoords(self) -> np.ndarray:
        return self.m_POICoords

    @property
    def extrinsicMat(self) -> np.ndarray:
        return np.concatenate([self.rotation, self.translation], axis=1)
    
    @property
    def projectionMat(self) -> np.ndarray:
        return np.dot(self.intrinsicMat, self.extrinsicMat)

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

def RefineCharacterisations(vc1: VisionConfig, vc2: VisionConfig, worldCoords):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT

    reprojErr, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        worldCoords, vc1.posePOICoords, vc2.posePOICoords,
        vc1.intrinsicMat, vc1.distortMat, vc2.intrinsicMat, vc2.distortMat,
        None, flags=flags)
    
    newVC1 = VisionConfig(vc1.rotation, vc1.translation,
        K1, D1, vc1.reprojErr, vc1.targetResolution, vc1.posePOICoords, vc1.boardPoses)
    
    # Update second device's world position
    newVC2 = VisionConfig(R @ vc1.rotation, R @ vc1.translation + T,
        K2, D2, vc2.reprojErr, vc2.targetResolution, vc2.posePOICoords, vc2.boardPoses)

    # Update vision configs of devices
    return newVC1, newVC2, reprojErr