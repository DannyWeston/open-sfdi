from __future__ import annotations

import numpy as np
import cv2

from abc import ABC, abstractmethod
from . import image, utils, colour

def show_pois(img, poi_count, poi_coords, size=None, colour_by=None):
    winName = f"Detected POIs"

    with utils.ProcessingContext.UseGPU(False):
        xp = utils.ProcessingContext().xp

        img = utils.ToContext(xp, img)
        if len(img.shape) == 2: 
            img = image.ExpandN(img, 3)

        poi_coords = utils.ToContext(xp, poi_coords)
        
        if colour_by is not None:
            colour_by = utils.ToContext(xp, colour_by)
            colour_by = image.Normalise(colour_by)

            # Sort by individual reprojection errors
            poi_coords = poi_coords[np.argsort(colour_by)]
            poi_coords = poi_coords.astype(np.uint16)

            for i in range(len(poi_coords)):
                # colour = (0.0, 1.0, 0.0) if reprojErrs[i] < 0 else (0.0, 0.0, 1.0)
                colour = (0.0, 1.0 - colour_by[i], float(colour_by[i]))
                img = cv2.circle(img, poi_coords[i], 3, colour, -1)
        
        else:
            # for i, (x, y) in enumerate(poiCoords):
            #     img = cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1, cv2.FILLED)

            img = image.ToInt(img)
            img = cv2.drawChessboardCorners(img.copy(), poi_count, poi_coords, True)

        image.show_img(img, winName, size=size)


# Characterisation Boards

class CharacterisationBoard(ABC):
    def __init__(self, poi_count):
        self._poi_count = poi_count
        
        self._debug = False
    
    @property
    def debug(self):
        return self._debug
    
    @debug.setter
    def debug(self, value):
        self._debug = value

    @property
    def poi_count(self):
        return self._poi_count
    
    @abstractmethod
    def find_pois(self, img: np.ndarray):
        raise NotImplementedError
    
    @abstractmethod
    def get_poi_coords(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_board_centre_coords(self) -> np.ndarray:
        raise NotImplementedError

class Checkerboard(CharacterisationBoard):
    def __init__(self, poi_count=(7, 10), square_size=(0.018, 0.018)):
        super().__init__(poi_count)

        self._square_size = square_size

    @property
    def square_size(self):
        return self._square_size

    def find_pois(self, img: np.ndarray):
        # Change image to int if not already
        img = image.ToInt(img)

        # flags = cv2.CALIB_CB_EXHAUSTIVE
        # result, corners = cv2.findChessboardCornersSB(uint_img, cb_size, flags=flags)
        # if not result: return None

        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp

            img = utils.ToContext(xp, img)

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_EXHAUSTIVE 
            result, corners = cv2.findChessboardCorners(img, self.poi_count, flags=flags)

            if not result: return None

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
            corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), criteria)

        return corners.squeeze()

    def get_poi_coords(self, dtype=None):
        xp = utils.ProcessingContext().xp

        w, h = self.poi_count

        if dtype is None: dtype = xp.float32

        ys, xs = xp.mgrid[:h, :w].astype(dtype)
        xs *= self._square_size[0]
        ys *= self._square_size[1]
        zs = xp.zeros(w * h, dtype=dtype)

        return xp.vstack([xs.ravel(), ys.ravel(), zs]).T

    def get_board_centre_coords(self, dtype=None) -> np.ndarray:
        xp = utils.ProcessingContext().xp

        if dtype is None: dtype = xp.float32

        h, w = self._poi_count
        corners = xp.zeros((w * h, 3), dtype=dtype)
        corners[:, :2] = np.mgrid[:h, :w].T.reshape(-1, 2) * self._square_size

        return corners

class CircleBoard(CharacterisationBoard):
    def __init__(self, poi_count=(7, 10), spacing=(0.03, 0.03), circleDiameter=1.0, inverted=True, staggered=True, area_max=None, poi_mask=None):
        super().__init__(poi_count)

        self.m_CircleDiameter = circleDiameter
        self._spacing = spacing

        self._staggered = staggered
        self.m_Inverted = inverted

        self._poi_mask = poi_mask

        # Setup blob detector
        self.m_DetectorParams = cv2.SimpleBlobDetector_Params()

        self.m_DetectorParams.filterByCircularity = True
        self.m_DetectorParams.filterByConvexity = True
        self.m_DetectorParams.filterByInertia = True
        self.m_DetectorParams.minCircularity = 0.6
        self.m_DetectorParams.minConvexity = 0.75
        self.m_DetectorParams.minInertiaRatio = 0.5

        # Check if filter by area
        if area_max is not None:
            self.m_DetectorParams.filterByArea = True
            self.m_DetectorParams.maxArea = area_max

        else: self.m_DetectorParams.filterByArea = False

        # self.m_DetectorParams.minThreshold = int(poiMask * 255)
        # self.m_DetectorParams.maxThreshold = 255
        # self.m_DetectorParams.thresholdStep = 10

        # Filter by colour
        self.m_DetectorParams.filterByColor = True
        self.m_DetectorParams.blobColor = 255 if inverted else 0

    def find_pois(self, img):
        xp = utils.ProcessingContext().xp

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(self.m_DetectorParams)

        flags = cv2.CALIB_CB_CLUSTERING
        flags |= (cv2.CALIB_CB_ASYMMETRIC_GRID if self._staggered else cv2.CALIB_CB_SYMMETRIC_GRID)

        # Convert to single channel
        img = image.ToGrey(img)

        # Apply threshold mask if present
        if self._poi_mask is not None:
            mask = image.ThresholdMask(img, min=self._poi_mask[0], max=self._poi_mask[1])
            img = img * mask

        # Convert to uint
        img = image.ToInt(img)

        # Some jobs can only be run on the CPU (using np instead of cupy)... cv2.findCirclesGrid is stupid
        # To fix this you could implement a custom circle grid finder - I am too lazy and sleep deprived
        # Also, I don't know why, but opencv returns corners in x-y format rather than the usual y-x that
        # it uses throughout the rest of the shitty library... and contains a stupid redundant dimension!!!
        # This convention is used for anything to do with characterisation... NOTHING ELSE

        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp

            img = utils.ToContext(xp, img)

            # STUPID OPENCV DOESN'T SUPPORT UINT16s SO CONVERT IF NEEDED...
            # cv2.convertScaleAbs(img, dst=img, alpha=(xp.iinfo(xp.uint8).max / xp.iinfo(img.dtype).max))

            result, corners = cv2.findCirclesGrid(img, self.poi_count, blobDetector=detector, flags=flags)

            if not result: return None

            return xp.asarray(corners).reshape(-1, 2)

    def get_poi_coords(self, dtype=None):
        # Multiply by square width
        xp = utils.ProcessingContext().xp

        if dtype is None: dtype = xp.float32

        w, h = self.poi_count

        x_delta = self._spacing[0]
        y_delta = self._spacing[1]

        ys, xs = xp.mgrid[:h, :w].astype(dtype)
        xs *= x_delta
        ys *= (y_delta / 2) if self._staggered else y_delta
        
        return xp.vstack([xs.ravel(), ys.ravel(), xp.zeros(w * h, dtype=dtype)]).T

    def get_board_centre_coords(self) -> np.ndarray:
        xp = utils.ProcessingContext().xp

        # TODO: Implement mechanism for calculating non-staggered centres
        w, h = self._poi_count

        # Assume Z = 0 for a flat board
        # XYZ Format
        return xp.array([(w - 1) * self._spacing[1] / 2, (h - 0.5) * self._spacing[0] / 2, 0.0])


# Characterisation

class ZhangJointChar(utils.SerialisableMixin):
    def __init__(self, rotation, translation, reproj_err):
        self._reproj_err = reproj_err

        self._rotation = rotation
        self._translation = translation

    @property
    def reproj_err(self):
        return self._reproj_err

    @property
    def rotation(self):
        return self._rotation
    
    @property
    def translation(self):
        return self._translation
    
    def __str__(self):
        return f'<JointChar> Reprojection Error: {self._reproj_err:.4f}'

class ZhangChar(utils.SerialisableMixin):
    _exclude_fields = {'_pose_translations', '_pose_rotations'}
    # Rotation and Translation are the transformation from the characterisation board's frame to the camera frame.
    # The first pose (Oxyz1) is used to determine the camera offset from the board along with the Rotation (R) and Translation (T):
    # I.e: Cxyz1 =  (R   T) . Oxyz1
    #               (0   1)
    #
    # The list of in-calibration order board poses are described by: 
    # Oxyz_i =  (R_i   T_i) . Oxyz1 {i -> 0 .. poses used}
    #           (0     1)

    def __init__(self, rotation=None, translation=None, 
            intrinsic_mat=None, distort_mat=None, reproj_errs=None,
            sensor_size=None, focal_length=None, optical_centre=None,
            resolution=None, pose_poi_coords=None, board_poses=None,
        ):

        xp = utils.ProcessingContext().xp

        self._rotation = rotation
        self._translation = translation

        self._intrinsic_mat = intrinsic_mat
        
        if distort_mat is None:
            self._distort_mat = xp.zeros(shape=(5,), dtype=xp.float32)
        else:
            self._distort_mat = distort_mat

        self._reproj_errs = reproj_errs

        self._resolution = resolution

        self._pose_poi_coords = pose_poi_coords

        self._pose_rotations = None
        self._pose_translations = None

        self._board_poses = board_poses

        self._focal_length = focal_length
        self._sensor_size = sensor_size
        self._optical_centre = optical_centre

    @property
    def rotation(self):
        return self._rotation
    
    @property
    def translation(self):
        return self._translation
    
    @property
    def intrinsic_mat(self):
        return self._intrinsic_mat
    
    @property
    def distort_mat(self):
        return self._distort_mat
    
    @property
    def reproj_errs(self):
        return self._reproj_errs
    
    @property
    def resolution(self):
        return self._resolution
    
    @property
    def pose_poi_coords(self):
        return self._pose_poi_coords
    
    @property
    def board_poses(self):
        return self._board_poses
    
    @property
    def focal_length(self):
        return self._focal_length
    
    @property
    def sensor_size(self):
        return self._sensor_size
    
    @property
    def optical_centre(self):
        return self._optical_centre

    @property
    def extrinsic_mat(self):
        xp = utils.ProcessingContext().xp
        temp = xp.asarray(self.translation)
        return xp.concatenate([xp.asarray(self.rotation), temp[:, xp.newaxis]], axis=1)
    
    # def fov(self, resolution):
    #     xp = utils.ProcessingContext().xp
    #     w, h = resolution

    #     if self.intrinsic_mat is None:
    #         fx = self.focal_length[0] * (w / self.sensor_size[0])
    #         fy = self.focal_length[1] * (h / self.sensor_size[1])

    #     else:
    #         fx = self.intrinsic_mat[0, 0]
    #         fy = self.intrinsic_mat[1, 1]

    #     return (
    #         2 * xp.arctan2(w, 2 * fx),
    #         2 * xp.arctan2(h, 2 * fy),
    #     )

    @property
    def projection_mat(self):
        xp = utils.ProcessingContext().xp
        return xp.dot(xp.asarray(self.intrinsic_mat), self.extrinsic_mat)

    def execute(self, board: CharacterisationBoard, poiCoords, resolution, extraFlags=None):
        w, h = resolution

        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp

            boardCoords = utils.ToContext(xp, board.get_poi_coords())
            objectCoords = xp.repeat(boardCoords[xp.newaxis, ...], len(poiCoords), axis=0)

            poiCoords = utils.ToContext(xp, poiCoords)

            flags = 0

            if extraFlags: flags |= extraFlags

            # Check if an initial guess can be made
            if self.sensor_size and self.focal_length and self.optical_centre:
                flags |= cv2.CALIB_USE_INTRINSIC_GUESS

                fx, fy = self.focal_length
                sx, sy = self.sensor_size
                ox, oy = self.optical_centre

                kGuess = np.array([
                    [(w * fx) / sx, 0.0, (w - 1) * ox],
                    [0.0, (h * fy) / sy, (h - 1) * oy],
                    [0.0, 0.0, 1.0]
                ], dtype=xp.float32)
            
            else: kGuess = None

            _, self._intrinsic_mat, self._distort_mat, self._pose_rotations, self._pose_translations = cv2.calibrateCamera(
                objectCoords, poiCoords, (w, h), kGuess, utils.ToContext(xp, self.distort_mat), flags=flags
            )

            self._pose_rotations = xp.asarray(self._pose_rotations)
            self._pose_translations = xp.asarray(self._pose_translations)

            self._reproj_errs = self.calc_reproj_errs(objectCoords, poiCoords,
                self.intrinsic_mat, self.distort_mat, self._pose_rotations, self._pose_translations)

            self._rotation = cv2.Rodrigues(self._pose_rotations[0])[0]
            self._translation = self._pose_translations[0].squeeze()

            self._resolution = resolution
            self._pose_poi_coords = poiCoords

            M0 = utils.TransMat(self.rotation, self.translation)

            boardPoses = [xp.eye(4, 4)]
            for i in range(1, len(objectCoords)):
                Mi = utils.TransMat(cv2.Rodrigues(self._pose_rotations[i])[0], self._pose_translations[i].squeeze())
                boardPoses.append(xp.linalg.inv(Mi) @ M0)

            self._board_poses = xp.asarray(boardPoses)

        return self._reproj_errs

    def undistort_img(self, img_data):
        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp

            if (self.intrinsic_mat is None) or (self.distort_mat is None): 
                return img_data
            
            # cv2 needs numpy..
            img_data = utils.ToContext(xp, img_data)
            img_data = cv2.undistort(img_data, self.intrinsic_mat, self.distort_mat)  

        xp = utils.ProcessingContext().xp

        return xp.asarray(img_data)
    
    def undistort_points(self, pois):
        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp

            if (self.intrinsic_mat is None) or (self.distort_mat is None): 
                return pois
            
            pois = utils.ToContext(xp, pois)
            cv2.undistortPoints(pois, self.intrinsic_mat, self.distort_mat, dst=pois, P=self.intrinsic_mat)

        return xp.asarray(pois).reshape(-1, 2)

    def joint_char(self, other: ZhangChar, board: CharacterisationBoard) -> ZhangJointChar:
        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp
            
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

            boardCoords = board.get_poi_coords()
            obj_coords = xp.repeat(boardCoords[xp.newaxis, ...], len(self._pose_poi_coords), axis=0)

            reproj_rms, self._intrinsic_mat, self._distort_mat, other._intrinsic_mat, other._distort_mat, R, T, E, F = \
                cv2.stereoCalibrate(obj_coords, self.pose_poi_coords, other.pose_poi_coords,
                self.intrinsic_mat, self.distort_mat, other.intrinsic_mat, other.distort_mat,
                self.resolution, criteria=criteria, flags=flags
            )
            
            # Other reproj error (need to use R and T)
            for i in range(len(other._pose_rotations)):
                _, R2, T2 = cv2.solvePnP(
                    obj_coords[i], self.pose_poi_coords[i], self.intrinsic_mat, self.distort_mat,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                R2 = cv2.Rodrigues(R2)[0]
                other._pose_rotations[i] = cv2.Rodrigues(R @ R2)[0]
                other._pose_translations[i] = R @ T2 + T.reshape(3, 1)

            return ZhangJointChar(R, T, reproj_rms)

    def calc_reproj_errs(self, obj_coords, poi_coords, intrinsic_mat, dist_mat, rotations, translations):
        reprojErrors = np.empty(shape=poi_coords.shape[:2], dtype=np.float32)
        
        for i in range(len(obj_coords)):
            projected_points, _ = cv2.projectPoints(obj_coords[i], rotations[i], translations[i], intrinsic_mat, dist_mat)
            
            # Calculate Euclidean distance for each point
            errors = np.linalg.norm(projected_points.squeeze() - poi_coords[i], axis=1).flatten()

            reprojErrors[i] = errors

        return np.asarray(reprojErrors)
    
    def __str__(self):
        xp = utils.ProcessingContext().xp

        v = f'<Char>'

        if self.reproj_errs is not None:
            reproj_rms = xp.sqrt(xp.mean((self._reproj_errs ** 2).flatten()))
            v += f' Reprojection Error: {reproj_rms:.4f}'

        return v
    

# Interfaces

class ICharable(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def char(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def resolution(self):
        raise NotImplementedError
        