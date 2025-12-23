import numpy as np
import cv2

from abc import ABC, abstractmethod
from .. import image, utils

def ShowPOIs(img, poiCoords, size=None, colourBy=None):
    winName = f"Detected POIs"

    if len(img.shape) == 2: img = image.ExpandN(img, 3)
    img = utils.ToNumpy(img)

    poiCoords = utils.ToNumpy(poiCoords)
    
    if colourBy is not None:
        with utils.ProcessingContext.UseGPU(False):
            colourBy = utils.ToNumpy(colourBy)
            colourBy = image.Normalise(colourBy)

        # Sort by individual reprojection errors
        poiCoords = poiCoords[np.argsort(colourBy)]
        poiCoords = poiCoords.astype(np.uint16)

        for i in range(len(poiCoords)):
            # colour = (0.0, 1.0, 0.0) if reprojErrs[i] < 0 else (0.0, 0.0, 1.0)
            colour = (0.0, 1.0 - colourBy[i], float(colourBy[i]))
            img = cv2.circle(img, poiCoords[i], 3, colour, -1)
    
    else:
        # for i, (x, y) in enumerate(poiCoords):
        #     img = cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1, cv2.FILLED)

        img = image.ToInt(img)
        img = cv2.drawChessboardCorners(img, (4, 13), poiCoords, True)

    image.Show(img, winName, size=size)

# CalibrationBoard

class CalibrationBoard(ABC):
    def __init__(self):
        self.m_Debug = False
    
    @property
    def debug(self):
        return self.m_Debug
    
    @debug.setter
    def debug(self, value):
        self.m_Debug = value

    @abstractmethod
    def FindPOIS(self, img: np.ndarray):
        raise NotImplementedError
    
    @abstractmethod
    def GetPOICoords(self):
        raise NotImplementedError
    
    @abstractmethod
    def GetBoardCentreCoords(self) -> np.ndarray:
        raise NotImplementedError

class Checkerboard(CalibrationBoard):
    def __init__(self, squareWidth=0.018, poiCount=(7, 10)):
        super().__init__()

        self.m_POICount = poiCount
        self.m_SquareWidth = squareWidth

    def FindPOIS(self, img: np.ndarray):
        # Change image to int if not already
        uint_img = image.ToInt(img)

        # flags = cv2.CALIB_CB_EXHAUSTIVE
        # result, corners = cv2.findChessboardCornersSB(uint_img, cb_size, flags=flags)
        # if not result: return None

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_EXHAUSTIVE
        result, corners = cv2.findChessboardCorners(uint_img, self.m_POICount, flags=flags)

        if not result: return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(uint_img, corners, (15, 15), (-1, -1), criteria)

        return corners.squeeze()

    def GetPOICoords(self, dtype=None):
        xp = utils.ProcessingContext().xp

        h, w = self.m_POICount

        if dtype is None: dtype = xp.float32

        xs, ys = xp.meshgrid(xp.arange(h), xp.arange(w))
        xs = xs.astype(dtype) * self.m_SquareWidth
        ys = ys.astype(dtype) * self.m_SquareWidth
            
        zs = xp.zeros((h * w), dtype=dtype)
        pois = xp.vstack([xs.ravel(), ys.ravel(), zs]).T

        return pois

    def GetBoardCentreCoords(self, dtype=None) -> np.ndarray:
        xp = utils.ProcessingContext().xp

        if dtype is None: dtype = xp.float32

        h, w = self.m_POICount
        corners = xp.zeros((w * h, 3), dtype=dtype)
        corners[:, :2] = np.mgrid[:h, :w].T.reshape(-1, 2) * self.m_SquareWidth

        return corners

class CircleBoard(CalibrationBoard):
    def __init__(self, circleSpacing=(0.03, 0.03), circleDiameter=1.0, poiCount=(4, 13), inverted=True, staggered=True, areaHint=None, poiMask=None):
        super().__init__()
        # TODO: Maybe make separate immutable classes for staggered and unstaggered?
        
        self.m_POICount = poiCount
        self.m_CircleDiameter = circleDiameter
        self.m_CircleSpacing = circleSpacing

        self.m_Staggered = staggered
        self.m_Inverted = inverted

        self.m_POIMask = poiMask

        # Setup blob detector
        self.m_DetectorParams = cv2.SimpleBlobDetector_Params()

        # Check if filter by area
        if areaHint is not None:
            # TODO: Implement polynomial to find rough relationship between circle sizes and resolution
            self.m_DetectorParams.filterByArea = True
            self.m_DetectorParams.minArea = areaHint[0]
            self.m_DetectorParams.maxArea = areaHint[1]

        else: self.m_DetectorParams.filterByArea = False

        # self.m_DetectorParams.minThreshold = int(poiMask * 255)
        # self.m_DetectorParams.maxThreshold = 255
        # self.m_DetectorParams.thresholdStep = 10

        # Filter by colour
        self.m_DetectorParams.filterByColor = True
        self.m_DetectorParams.blobColor = 255 if inverted else 0

    def FindPOIS(self, img):
        xp = utils.ProcessingContext().xp

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(self.m_DetectorParams)

        flags = cv2.CALIB_CB_CLUSTERING
        flags += cv2.CALIB_CB_ASYMMETRIC_GRID if self.m_Staggered else cv2.CALIB_CB_SYMMETRIC_GRID

        img = image.ToGrey(img)

        # Apply threshold mask if present
        if self.m_POIMask is not None:
            mask = image.ThresholdMask(img, min=self.m_POIMask[0], max=self.m_POIMask[1])
            img = img * mask

        # Convert to uint
        img = image.ToInt(img)

        # Some jobs can only be run on the CPU (using np instead of cupy)... cv2.findCirclesGrid is stupid
        # To fix this you could implement a custom circle grid finder - I am too lazy and sleep deprived
        # Also, I don't know why, but opencv returns corners in x-y format rather than the usual y-x that
        # it uses throughout the rest of the shitty library... and contains a stupid redundant dimension!!!
        # This convention is used for anything to do with characterisation... NOTHING ELSE

        img = utils.ToNumpy(img)

        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp

            # STUPID OPENCV DOESN'T SUPPORT UINT16s SO CONVERT IF NEEDED...
            img = cv2.convertScaleAbs(img, alpha=(xp.iinfo(xp.uint8).max / xp.iinfo(img.dtype).max))

            result, corners = cv2.findCirclesGrid(img, self.m_POICount, blobDetector=detector, flags=flags)

        xp = utils.ProcessingContext().xp

        # Back in old xp context now
        if not result: return None

        return xp.asarray(corners).reshape(-1, 2)

    def GetPOICoords(self, dtype=None):
        # Multiply by square width
        xp = utils.ProcessingContext().xp

        if dtype is None: dtype = xp.float32

        h, w = self.m_POICount

        xDelta = self.m_CircleSpacing[1] / 2
        yDelta = self.m_CircleSpacing[0]

        # Setup checkerboard world coordinates
        if self.m_Staggered:
            xs, ys = xp.mgrid[:w, :h].astype(dtype)
            xs *= xDelta

            ys *= yDelta
            ys[1::2] += yDelta / 2
            
            zs = xp.zeros((h * w), dtype=dtype)
            circleCentres = xp.vstack([xs.ravel(), ys.ravel(), zs]).T
        else:
            circleCentres = xp.zeros((w * h, 3), dtype=dtype)
            circleCentres[:, :2] = xp.mgrid[:w, :h].T.reshape(-1, 2) * self.m_CircleSpacing[0]

        return circleCentres

    def GetBoardCentreCoords(self) -> np.ndarray:
        xp = utils.ProcessingContext().xp

        # TODO: Implement mechanism for calculating non-staggered centres
        h, w = self.m_POICount

        # Assume Z = 0 for a flat board
        # XYZ Format
        return xp.array([(w - 1) * self.m_CircleSpacing[1] / 2, (h - 0.5) * self.m_CircleSpacing[0] / 2, 0.0])

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
        
        if self.distortMat is None:
            self.distortMat = np.zeros(shape=(5,), dtype=np.float32)

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

    def Calculate(self, board: CalibrationBoard, poiCoords, resolution, extraFlags=None):
        h, w = resolution

        boardCoords = utils.ToNumpy(board.GetPOICoords())
        objectCoords = np.repeat(boardCoords[np.newaxis, ...], len(poiCoords), axis=0)

        poiCoords = utils.ToNumpy(poiCoords)

        flags = 0

        if extraFlags: flags += extraFlags

        # Check if an initial guess can be made
        if self.sensorSizeGuess and self.focalLengthGuess and self.opticalCentreGuess:
            fx, fy = self.focalLengthGuess
            sx, sy = self.sensorSizeGuess
            ox, oy = self.opticalCentreGuess

            kGuess = np.array([
                [(w * fx) / sx, 0.0, (w - 1) * ox],
                [0.0, (h * fy) / sy, (h - 1) * oy],
                [0.0, 0.0, 1.0]
            ])

            
        # "proIntri": np.array([[2188.80000000000, 0,  912],
        #                       [0,  2188.80000000000, 1140],
        #                       [0,  0,  1 ]], np.float64),

        # proIntri [[1104.279 0.0 438.698]
        #           [0.0 2206.32 1204.100]
        #           [0.0 0.0 1.0]]

            flags += cv2.CALIB_USE_INTRINSIC_GUESS

            self.reprojErr, self.intrinsicMat, self.distortMat, R, T = cv2.calibrateCamera(
                objectCoords, poiCoords, (w, h), kGuess, self.distortMat, flags=flags
            )
        
        else: 
            self.reprojErr, self.intrinsicMat, self.distortMat, R, T = cv2.calibrateCamera(
                objectCoords, poiCoords, (w, h), None, self.distortMat, flags=flags
            )

        R = np.asarray(R)

        reprojErrs, rmsError = self.ReprojectionErrors(objectCoords, poiCoords, 
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

    def Undistort(self, imgData):
        xp = utils.ProcessingContext().xp

        if (self.intrinsicMat is None) or (self.distortMat is None): 
            return imgData
        
        # cv2 needs numpy..
        imgData = utils.ToNumpy(imgData)
        imgData = cv2.undistort(imgData, self.intrinsicMat, self.distortMat, None, self.intrinsicMat)  

        return xp.asarray(imgData)
    
    def UndistortPoints(self, pois):
        xp = utils.ProcessingContext().xp

        if (self.intrinsicMat is None) or (self.distortMat is None): 
            return pois
        
        pois = utils.ToNumpy(pois)
        pois2 = cv2.undistortPoints(pois, self.intrinsicMat, self.distortMat, P=self.intrinsicMat)

        return xp.asarray(pois2).reshape(-1, 2)

    def ReprojectionErrors(self, objectCoords, poiCoords, intrinsicMat, distMat, rotations, translations):
        reprojErrors = []

        xErrors = []
        yErrors = []
        
        for i in range(len(objectCoords)):
            projectedPoints, _ = cv2.projectPoints(objectCoords[i], 
                rotations[i], translations[i], intrinsicMat, distMat)
            
            projectedPoints = projectedPoints.squeeze()
            xPoints = projectedPoints.copy()
            xPoints[:, 1] = 0.0

            yPoints = projectedPoints.copy()
            yPoints[:, 0] = 0.0

            xPois = poiCoords[i].copy()
            xPois[:, 1] = 0.0

            yPois = poiCoords[i].copy()
            yPois[:, 0] = 0.0
            
            # Calculate Euclidean distance for each point
            errors = np.linalg.norm(projectedPoints - poiCoords[i], axis=1)
            # print(f"Image {i} Reprojection Error: Min-max ({errors.min()}, {errors.max()})")

            x = np.linalg.norm(xPoints - xPois, axis=1)
            # print(f"X-reprojection Error: Min-max ({x.min()}, {x.max()})")

            y = np.linalg.norm(yPoints - yPois, axis=1)
            # print(f"Y-reprojection Error: Min-max ({y.min()}, {y.max()})")
            # print()

            reprojErrors.append(errors)

        reprojErrors = np.asarray(reprojErrors)
        rmsError = np.sqrt(np.mean((reprojErrors ** 2).flatten()))

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

def RefineCharacterisations(vc1: Characterisation, vc2: Characterisation, board: CalibrationBoard):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS

    boardCoords = utils.ToNumpy(board.GetPOICoords())
    objectCoords = np.repeat(boardCoords[np.newaxis, ...], len(vc1.poiCoords), axis=0)

    reprojErr, vc1.intrinsicMat, vc1.distortMat, vc2.intrinsicMat, vc2.distortMat, R, T, E, F = cv2.stereoCalibrate(
        objectCoords, vc1.poiCoords, vc2.poiCoords,
        vc1.intrinsicMat, vc1.distortMat, vc2.intrinsicMat, vc2.distortMat,
        None, flags=flags)
    
    # Update second device's world position
    vc2.rotation = R @ vc1.rotation
    vc2.translation = R @ vc1.translation + T

    # Update vision configs of devices
    return reprojErr