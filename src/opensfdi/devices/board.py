import numpy as np
import cv2

from abc import ABC, abstractmethod
from .. import image

# CalibrationBoard

class CalibrationBoard(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def FindPOIS(self, img: np.ndarray):
        raise NotImplementedError
    
    def GetPOICoords(self):
        raise NotImplementedError

class Checkerboard(CalibrationBoard):
    def __init__(self, square_width=0.018, poi_count=(10, 7)):
        self.__poi_count = poi_count
        self.__square_width = square_width

        # Multiply by square width
        w, h = self.__poi_count
        self.__cb_corners = np.zeros((w * h, 3), np.float32)
        self.__cb_corners[:, :2] = np.mgrid[:w, :h].T.reshape(-1, 2) * self.__square_width

    def FindPOIS(self, img: np.ndarray):
        # Change image to int if not already
        uint_img = image.ToU8(img)

        # flags = cv2.CALIB_CB_EXHAUSTIVE
        # result, corners = cv2.findChessboardCornersSB(uint_img, cb_size, flags=flags)
        # if not result: return None

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_EXHAUSTIVE
        result, corners = cv2.findChessboardCorners(uint_img, self.__poi_count, flags=flags)

        if not result: return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(uint_img, corners, (15, 15), (-1, -1), criteria)

        return corners.squeeze()

    def GetPOICoords(self):
        return self.__cb_corners.copy()

class CircleBoard(CalibrationBoard):
    
    def __init__(self, circle_diameter=1.0, circleSpacing=1.0, poiCount=(10, 7), inverted=True, staggered=True, areaHint=None):
        self.__poi_count = poiCount
        self.__diameter = circle_diameter
        self.__spacing = circleSpacing

        self.__staggered = staggered

        # Setup blob detector
        self.__detector_params = cv2.SimpleBlobDetector_Params()

        if areaHint is not None:
            # 480x270   :   (100, 400)
            # 640x360   :   (200, 1000)
            # 960x540   :   (500, 2000)
            # 1440x810  :   (1250, 5000)
            # 1920x1080 :   (2500, 13000)
            # 2560x1440 :   (5000, 20000)
            # 3840x2160 :   (7000, 28000)

            self.__detector_params.filterByArea = True
            self.__detector_params.minArea = areaHint[0]
            self.__detector_params.maxArea = areaHint[1]

        self.__detector_params.blobColor = 255 if inverted else 0

        # Multiply by square width
        w, h = self.__poi_count

        # Setup checkerboard world coordinates
        if self.__staggered:
            self.__circle_centres = np.zeros((w * h, 3), dtype=np.float32)
            for i in range(h):
                    for j in range(w):
                        self.__circle_centres[i * w + j, :2] = [(2 * j + i % 2) * self.__spacing / 2, i * self.__spacing / 2]
        else:
            self.__circle_centres = np.zeros((w * h, 3), np.float32)
            self.__circle_centres[:, :2] = np.mgrid[:w, :h].T.reshape(-1, 2) * self.__spacing

    def FindPOIS(self, img: np.ndarray):
        # Mask the image
        img = image.ThresholdMask(img, threshold=0.1)

        # Convert image to uint datatype
        uint_img = image.ToU8(img)

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(self.__detector_params)

        flags = (cv2.CALIB_CB_ASYMMETRIC_GRID if self.__staggered else cv2.CALIB_CB_SYMMETRIC_GRID)

        result, corners = cv2.findCirclesGrid(uint_img, self.__poi_count, blobDetector=detector, flags=flags)

        if not result: return None

        # DEBUG: Show the corners on the image
        # corners_img = cv2.drawChessboardCorners(uint_img, self.__poi_count, corners, result)
        # image.show_image(corners_img)

        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # corners = cv2.cornerSubPix(uint_img, corners, (15, 15), (-1, -1), criteria)

        return corners.squeeze()

    def GetPOICoords(self):
        return self.__circle_centres.copy()