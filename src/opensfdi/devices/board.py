import numpy as np
import cv2

from abc import ABC, abstractmethod

from ..utils import AlwaysNumpy

from .. import image


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
    def __init__(self, circleSpacing=(0.03, 0.03), circleDiameter=1.0, poiCount=(4, 13), inverted=True, staggered=True, areaHint=None):
        super().__init__()
        # TODO: Maybe make separate immutable classes for staggered and unstaggered?
        
        self.m_POICount = poiCount
        self.m_CircleDiameter = circleDiameter
        self.m_CircleSpacing = circleSpacing

        self.m_Staggered = staggered
        self.m_Inverted = inverted

        # Setup blob detector
        self.m_DetectorParams = cv2.SimpleBlobDetector_Params()

        if areaHint is not None:
            # TODO: Implement polynomial to find rough relationship between circle sizes and resolution
            # 480x270   :   (100, 400)
            # 640x360   :   (200, 1000)
            # 960x540   :   (500, 2000)
            # 1440x810  :   (1250, 5000)
            # 1920x1080 :   (2500, 13000)
            # 2560x1440 :   (5000, 20000)
            # 3840x2160 :   (7000, 28000)

            self.m_DetectorParams.filterByArea = True
            self.m_DetectorParams.minArea = areaHint[0]
            self.m_DetectorParams.maxArea = areaHint[1]

        self.m_DetectorParams.blobColor = 255 if inverted else 0

    def FindPOIS(self, img):
        img = AlwaysNumpy(img)

        # Mask the image
        img = image.ThresholdMask(img, threshold=0.1)

        # Convert image to uint datatype
        uint_img = image.ToU8(img)

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(self.m_DetectorParams)

        flags = (cv2.CALIB_CB_ASYMMETRIC_GRID if self.m_Staggered else cv2.CALIB_CB_SYMMETRIC_GRID)

        result, corners = cv2.findCirclesGrid(uint_img, self.m_POICount, blobDetector=detector, flags=flags)

        if not result:
            if self.debug:
                print("Failed to detect POIs")
                image.Show(uint_img, size=(2100, 1400))

            return None
        
        corners = corners.squeeze()
            
        if self.debug:
            print("Successfully detected POIs")
            debugImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
            debugImage = cv2.drawChessboardCorners(debugImage, self.m_POICount, corners, True)
            debugImage = cv2.circle(debugImage, corners[0].astype(int), 10, (255, 0, 0), -1)
            image.Show(debugImage, size=(2100, 1400))

        return corners

    def GetPOICoords(self):
        # Multiply by square width
        h, w = self.m_POICount

        # Setup checkerboard world coordinates
        if self.m_Staggered:
            xs, ys = np.mgrid[:w, :h].astype(np.float32)
            xs *= self.m_CircleSpacing[1] / 2

            ys *= self.m_CircleSpacing[0]
            ys[1::2] += self.m_CircleSpacing[0] / 2
            
            zs = np.zeros((h * w), dtype=np.float32)

            circleCentres = np.vstack([xs.ravel(), ys.ravel(), zs]).T
        else:
            circleCentres = np.zeros((w * h, 3), np.float32)
            circleCentres[:, :2] = np.mgrid[:w, :h].T.reshape(-1, 2) * self.m_CircleSpacing[0]

        return circleCentres

    def GetBoardCentreCoords(self) -> np.ndarray:
        # TODO: Implement mechanism for calculating non-staggered centres
        h, w = self.m_POICount

        # Assume Z = 0 for a flat board
        # XYZ Format
        return np.array([(w - 1) * self.m_CircleSpacing[1] / 2, (h - 0.5) * self.m_CircleSpacing[0] / 2, 0.0])