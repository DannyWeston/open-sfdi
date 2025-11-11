import numpy as np
import cv2

from abc import ABC, abstractmethod

from .. import utils

class Characterisation:
    
    # Rotation and Translation are the transformation from the characterisation board's frame to the camera frame.
    # The first pose (Oxyz1) is used to determine the camera offset from the board along with the Rotation (R) and Translation (T):
    # I.e: Cxyz1 =  (R   T) . Oxyz1
    #               (0   1)
    #
    # The list of in-calibration order board poses are described by: 
    # Oxyz_i =  (R_i   T_i) . Oxyz1 {i -> 0 .. poses used}
    #           (0     1)

    def __init__(self, rotation=None, translation=None, 
            intrinsicMat=None, distortMat=None, reprojErr=None,
            sensorSizeGuess=None, focalLengthGuess=None, opticalCentreGuess=None,
            targetResolution=None, posePOICoords=None, boardPoses=None,
        ):

        self.rotation = rotation
        self.translation = translation

        self.intrinsicMat = intrinsicMat
        self.distortMat = distortMat
        self.reprojErr = reprojErr

        self.targetResolution = targetResolution

        self.poiCoords = posePOICoords

        self.boardPoses = boardPoses

        self.focalLengthGuess = focalLengthGuess
        self.sensorSizeGuess = sensorSizeGuess
        self.opticalCentreGuess = opticalCentreGuess

    @property
    def extrinsicMat(self) -> np.ndarray:
        return np.concatenate([self.rotation, self.translation[:, np.newaxis]], axis=1)
    
    @property
    def projectionMat(self) -> np.ndarray:
        return np.dot(self.intrinsicMat, self.extrinsicMat)

    def Calculate(self, resolution, objectCoords, poiCoords, extraFlags=None):
        h, w = resolution

        objectCoords = utils.ToNumpy(objectCoords)
        poiCoords = utils.ToNumpy(poiCoords)

        flags = 0

        # Check if an initial guess can be made
        if self.sensorSizeGuess and self.focalLengthGuess and self.opticalCentreGuess:
            fx, fy = self.focalLengthGuess
            sx, sy = self.sensorSizeGuess
            ox, oy = self.opticalCentreGuess

            kGuess = np.array([
                [w * fx / sx, 0.0, (w - 1) * ox],
                [0.0, h * fy / sy, (h - 1) * oy],
                [0.0, 0.0, 1.0]
            ])

            flags += cv2.CALIB_USE_INTRINSIC_GUESS
        
        else: kGuess = None

        if extraFlags: 
            flags += extraFlags

        self.reprojErr, self.intrinsicMat, self.distortMat, R, T = cv2.calibrateCamera(
            objectCoords, poiCoords, (w, h), kGuess, self.distortMat, flags=flags
        )

        R = np.asarray(R)

        reprojErrs, rmsError = self.__ReprojectionErrors(objectCoords, poiCoords, 
            self.intrinsicMat, self.distortMat, R, T)

        self.rotation = cv2.Rodrigues(R[0])[0]
        self.translation = T[0].squeeze()

        self.targetResolution = resolution
        self.poiCoords = poiCoords

        M0 = utils.TransMat(self.rotation, self.translation)

        boardPoses = [np.eye(4, 4)]
        for i in range(1, len(objectCoords)):
            Mi = utils.TransMat(cv2.Rodrigues(R[i])[0], T[i].squeeze())
            boardPoses.append(np.linalg.inv(Mi) @ M0)

        self.boardPoses = np.asarray(boardPoses)

        return reprojErrs, rmsError

    def __ReprojectionErrors(self, objectCoords, poiCoords, intrinsicMat, distMat, rotations, translations):
        reprojErrors = []
        
        for i in range(len(objectCoords)):
            projectedPoints, _ = cv2.projectPoints(objectCoords[i], 
                rotations[i], translations[i], intrinsicMat, distMat)
            
            projectedPoints = projectedPoints.squeeze()
            
            # Calculate Euclidean distance for each point
            errors = np.linalg.norm(projectedPoints - poiCoords[i], axis=1)
            
            print(f"Image {i} Reprojection Error: Min-max ({errors.min()}, {errors.max()})")
            reprojErrors.append(errors)

        reprojErrors = np.asarray(reprojErrors)
        rmsError = np.sqrt(np.mean((reprojErrors ** 2).flatten()))
        print(f"RMS reprojection error {rmsError}")

        return reprojErrors, rmsError

class ICharacterisable(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def characterisation(self) -> Characterisation:
        raise NotImplementedError

    @abstractmethod
    def Characterise(self, resolution, worldCoords, pixelCoords):
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



def RefineCharacterisations(vc1: Characterisation, vc2: Characterisation, objectCoords):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    objectCoords = utils.ToNumpy(objectCoords)

    reprojErr, vc1.intrinsicMat, vc1.distortMat, vc2.intrinsicMat, vc2.distortMat, R, T, E, F = cv2.stereoCalibrate(
        objectCoords, vc1.poiCoords, vc2.poiCoords,
        vc1.intrinsicMat, vc1.distortMat, vc2.intrinsicMat, vc2.distortMat,
        None, flags=flags, criteria=criteria)
    
    # Update second device's world position
    vc2.rotation = R @ vc1.rotation
    vc2.translation = R @ vc1.translation + T

    # Update vision configs of devices
    return reprojErr