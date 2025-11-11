import numpy as np
import cv2

from abc import ABC, abstractmethod

from .. import image, utils


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
        uint_img = image.ToU8(img)

        # flags = cv2.CALIB_CB_EXHAUSTIVE
        # result, corners = cv2.findChessboardCornersSB(uint_img, cb_size, flags=flags)
        # if not result: return None

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_EXHAUSTIVE
        result, corners = cv2.findChessboardCorners(uint_img, self.m_POICount, flags=flags)

        if not result: return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(uint_img, corners, (15, 15), (-1, -1), criteria)

        return corners.squeeze()

    def GetPOICoords(self):
        return self.__cb_corners.copy()

    def GetBoardCentreCoords(self) -> np.ndarray:
        # Multiply by square width
        h, w = self.m_POICount
        corners = np.zeros((w * h, 3), np.float32)
        corners[:, :2] = np.mgrid[:h, :w].T.reshape(-1, 2) * self.m_SquareWidth

        return corners

class CircleBoard(CalibrationBoard):
    def __init__(self, circleSpacing=(0.03, 0.03), circleDiameter=1.0, poiCount=(4, 13), inverted=True, staggered=True, areaHint=None, poiMask=(0.1, 0.9)):
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
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(self.m_DetectorParams)

        flags = cv2.CALIB_CB_CLUSTERING
        flags += cv2.CALIB_CB_ASYMMETRIC_GRID if self.m_Staggered else cv2.CALIB_CB_SYMMETRIC_GRID

        img = image.ToGrey(img)

        # Apply threshold mask if present
        if self.m_POIMask is not None:
            img = image.ThresholdMask(img, min=self.m_POIMask[0], max=self.m_POIMask[1])

        # Convert to numpy uint8 for cv2
        img = image.ToU8(img)

        # Some jobs can only be run on the CPU (using np instead of cupy)... cv2.findCirclesGrid is stupid
        # To fix this you could implement a custom circle grid finder - I am too lazy and sleep deprived
        # Also, I don't know why, but opencv returns corners in x-y format rather than the usual y-x that
        # it uses throughout the rest of the shitty library... and contains a stupid redundant dimension!!!
        # This convention is used for anything to do with characterisation... NOTHING ELSE

        img = utils.ToNumpy(img)

        result, corners = cv2.findCirclesGrid(img, self.m_POICount, blobDetector=detector, flags=flags)

        if not result: return None
        
        # Stepped out of forced CPU context, get valid device context back
        xp = utils.ProcessingContext().xp
        corners = xp.asarray(corners)

        return corners.reshape(-1, 2)

    def GetPOICoords(self):
        # Multiply by square width
        xp = utils.ProcessingContext().xp

        h, w = self.m_POICount

        xDelta = self.m_CircleSpacing[1] / 2
        yDelta = self.m_CircleSpacing[0]

        # Setup checkerboard world coordinates
        if self.m_Staggered:
            xs, ys = xp.mgrid[:w, :h].astype(xp.float32)
            xs *= xDelta

            ys *= yDelta
            ys[1::2] += yDelta / 2
            
            zs = xp.zeros((h * w), dtype=xp.float32)
            circleCentres = xp.vstack([xs.ravel(), ys.ravel(), zs]).T
        else:
            circleCentres = xp.zeros((w * h, 3), xp.float32)
            circleCentres[:, :2] = xp.mgrid[:w, :h].T.reshape(-1, 2) * self.m_CircleSpacing[0]

        return circleCentres

    def GetBoardCentreCoords(self) -> np.ndarray:
        xp = utils.ProcessingContext().xp

        # TODO: Implement mechanism for calculating non-staggered centres
        h, w = self.m_POICount

        # Assume Z = 0 for a flat board
        # XYZ Format
        return xp.array([(w - 1) * self.m_CircleSpacing[1] / 2, (h - 0.5) * self.m_CircleSpacing[0] / 2, 0.0])