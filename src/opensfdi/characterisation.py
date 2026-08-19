from __future__ import annotations

import numpy as np
import cv2

from abc import ABC, abstractmethod
from . import image, utils

def draw_pois(img: image.Image, poi_count, poi_coords, colour_by=None, numbered=False):
    with utils.ProcessingContext.UseGPU(False):
        xp = utils.ProcessingContext().xp

        img_data = img.raw_data.copy()

        img_data = utils.ToContext(xp, img_data)
        if len(img_data.shape) == 2:
            img_data = image.ExpandN(img_data, 3)

        poi_coords = utils.ToContext(xp, poi_coords)

        # draw_img = image.ToInt(img)
        draw_img = image.ToInt(img_data.copy())


        if colour_by is None:
            draw_img = cv2.drawChessboardCorners(draw_img, poi_count, poi_coords, True)
        else:
            colour_by = utils.ToContext(xp, colour_by)
            colour_by = image.Normalise(colour_by)

            # Sort by individual reprojection errors
            poi_coords = poi_coords.astype(xp.uint16)
            poi_coords = poi_coords[np.argsort(colour_by)]

            for i in range(len(poi_coords)):
                # colour = (0.0, 1.0, 0.0) if reprojErrs[i] < 0 else (0.0, 0.0, 1.0)
                colour = (0, 255 - colour_by[i], colour_by[i])
                draw_img = cv2.circle(draw_img, poi_coords[i], 3, colour, -1)

        if numbered:
            poi_coords = poi_coords.astype(xp.uint16)

            for i in range(len(poi_coords)):
                colour = (255, 0, 0)
                draw_img = cv2.putText(draw_img, str(i),  poi_coords[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image.Image(data=draw_img)

# Characterisation Boards

class NotCharacterisedException(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class CharacterisationBoard(ABC, utils.SerialisableMixin):
    def __init__(self, poi_count):
        self._poi_count = poi_count

    @property
    def poi_count(self):
        return self._poi_count
    
    @abstractmethod
    def find_pois(self, img: np.ndarray, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def get_poi_coords(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_board_centre_coords(self) -> np.ndarray:
        raise NotImplementedError

class Checkerboard(CharacterisationBoard):
    _exclude_fields = {"_corners_cache"}
    
    def __init__(self, poi_count=(7, 10), square_size=(0.018, 0.018)):
        super().__init__(poi_count)

        self._square_size = square_size

        self._corners_cache = np.empty(shape=(poi_count[0] * poi_count[1], 1, 2), dtype=np.float32)

    @property
    def square_size(self):
        return self._square_size

    def find_pois(self, img_data, sb=False, flags=0):
        img_data2 = image.ToInt(img_data)

        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp

            img_data2 = utils.ToContext(xp, img_data2)
            img_data2 = image.ToGrey(img_data2)

            if sb:
                result, self._corners_cache = cv2.findChessboardCornersSB(
                    img_data2,
                    self.poi_count,
                    None,
                    flags
                )
            else:
                result, self._corners_cache = cv2.findChessboardCorners(
                    img_data2,
                    self.poi_count,
                    None,
                    flags
                )

            if not result: return None
            
            return self._corners_cache.squeeze()

    def get_poi_coords(self, dtype=None):
        xp = utils.ProcessingContext().xp

        w, h = self.poi_count

        if dtype is None: dtype = xp.float32

        ys, xs = xp.mgrid[:h, :w].astype(dtype)
        xs *= self.square_size[0]
        ys *= self.square_size[1]
        zs = xp.zeros(w * h, dtype=dtype)

        return xp.vstack([xs.ravel(), ys.ravel(), zs]).T

    def get_board_centre_coords(self, dtype=None) -> np.ndarray:
        xp = utils.ProcessingContext().xp

        if dtype is None: dtype = xp.float32

        h, w = self._poi_count
        corners = xp.zeros((w * h, 3), dtype=dtype)
        corners[:, :2] = np.mgrid[:h, :w].T.reshape(-1, 2) * self._square_size

        return corners

class BlobDetector(utils.SerialisableMixin):
    _exclude_fields = {"_detector_params"}

    def __init__(self, circularity=(0.0, 1.0), convexity=(0.75, 1.0), inertiaRatio=(0.5, 1.0), inverted=True, poiMask=(0.0, 1.0), maxArea=-1.0):
        
        # Setup blob detector
        self._detector_params = cv2.SimpleBlobDetector_Params() # type: cv2.SimpleBlobDetector_Params

        # Circularity
        self._circularity = circularity
        self._detector_params.filterByCircularity = True
        self._detector_params.minCircularity = circularity[0]
        self._detector_params.maxCircularity = circularity[1]

        # Convexity
        self._convexity = convexity
        self._detector_params.filterByConvexity = True
        self._detector_params.minConvexity = convexity[0]
        self._detector_params.maxConvexity = convexity[1]

        # Inertia Ratio
        self._inertiaRatio = inertiaRatio
        self._detector_params.filterByInertia = True
        self._detector_params.minInertiaRatio = inertiaRatio[0]
        self._detector_params.maxInertiaRatio = inertiaRatio[1]

        # Filter by area
        self._detector_params.filterByArea = (0 < maxArea)
        if 0 < maxArea: self._detector_params.maxArea = maxArea

        self._poiMask = poiMask
        # self.m_DetectorParams.minThreshold = int(poiMask * 255)
        # self.m_DetectorParams.maxThreshold = 255
        # self.m_DetectorParams.thresholdStep = 10

        # Filter by colour
        self._inverted = inverted
        self._detector_params.filterByColor = True
        self._detector_params.blobColor = 255 if inverted else 0

    @property
    def circularity(self):
        return self._circularity

    @circularity.setter
    def circularity(self, value: tuple[float, float]):
        self._circularity = value
        self._detector_params.minCircularity = value[0]
        self._detector_params.maxCircularity = value[1]

    @property
    def convexity(self):
        return self._convexity

    @convexity.setter
    def convexity(self, value: tuple[float, float]):
        self._convexity = value
        self._detector_params.minConvexity = value[0]
        self._detector_params.maxConvexity = value[1]

    @property
    def inertiaRatio(self):
        return self._inertiaRatio

    @inertiaRatio.setter
    def inertiaRatio(self, value: tuple[float, float]):
        self._inertiaRatio = value
        self._detector_params.minInertiaRatio = value[0]
        self._detector_params.maxInertiaRatio = value[1]

    @property
    def poiMask(self):
        return self._poiMask

    @poiMask.setter
    def poiMask(self, value: tuple[float, float]):
        self._poiMask = value

    @property
    def inverted(self):
        return self._inverted

    @inverted.setter
    def inverted(self, value: bool):
        self._inverted = inverted
        self._detector_params.filterByColor = True
        self._detector_params.blobColor = 255 if value else 0

    @property
    def maxArea(self):
        return self._maxArea
    
    @maxArea.setter
    def maxArea(self, value: float):
        self._maxArea = value
        self._detector_params.filterByArea = (0.0 < value)
        if 0 < value: self._detector_params.maxArea = value

    def make_detector(self) -> cv2.SimpleBlobDetector:
        # Create a detector with the parameters
        return cv2.SimpleBlobDetector_create(self._detector_params)

class CircleBoard(CharacterisationBoard):
    def __init__(self, poi_count=(7, 10), spacing=(0.03, 0.03), diameter=1.0, staggered=True, blob_detector=None):
        super().__init__(poi_count)

        self._diameter = diameter
        self._spacing = spacing

        self._staggered = staggered

        self._blob_detector = blob_detector if blob_detector else BlobDetector()

    def find_pois(self, img, flags=0):
        xp = utils.ProcessingContext().xp

        flags |= cv2.CALIB_CB_CLUSTERING
        flags |= (cv2.CALIB_CB_ASYMMETRIC_GRID if self.staggered else cv2.CALIB_CB_SYMMETRIC_GRID)

        # Convert to single channel
        img = image.ToGrey(img)

        # Apply threshold mask if present
        if self.blob_detector.poiMask is not None:
            mask = image.ThresholdMask(img, *self.blob_detector.poiMask)
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
            result, corners = cv2.findCirclesGrid(
                img, self.poi_count,
                blobDetector=self.blob_detector.make_detector(), 
                flags=flags
            )

            if not result: return None

            return xp.asarray(corners).reshape(-1, 2)

    def get_poi_coords(self, dtype=None):
        # Multiply by square width
        xp = utils.ProcessingContext().xp

        if dtype is None: dtype = xp.float32

        w, h = self.poi_count

        x_delta, y_delta = self.spacing[1]

        ys, xs = xp.mgrid[:h, :w].astype(dtype)
        xs *= x_delta
        ys *= (y_delta / 2) if self.staggered else y_delta
        
        return xp.vstack([xs.ravel(), ys.ravel(), xp.zeros(w * h, dtype=dtype)]).T

    def get_board_centre_coords(self) -> np.ndarray:
        xp = utils.ProcessingContext().xp

        # TODO: Implement mechanism for calculating non-staggered centres
        w, h = self.poi_count

        # Assume Z = 0 for a flat board
        # XYZ Format
        return xp.array([(w - 1) * self.spacing[1] / 2, (h - 0.5) * self.spacing[0] / 2, 0.0])

    @property
    def blob_detector(self) -> BlobDetector:
        return self._blob_detector

    @property
    def spacing(self) -> tuple[float, float]:
        return self._spacing
    
    @property
    def diameter(self) -> float:
        return self._diameter
    
    @property
    def staggered(self) -> bool:
        return self._staggered

# Characterisation

class ZhangChar(utils.SerialisableMixin):
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
            rotation_to_other=None, translation_to_other=None,
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

        self._rotation_to_other = rotation_to_other
        self._translation_to_other = translation_to_other

        self._board_poses = board_poses

        self._focal_length = focal_length
        self._sensor_size = sensor_size
        self._optical_centre = optical_centre

    @property
    def exclude_fields(self):
        return super().exclude_fields.union({'_pose_translations', '_pose_rotations'})

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
    def rotation_to_other(self):
        return self._rotation_to_other
    
    @property
    def translation_to_other(self):
        return self._translation_to_other

    @property
    def projection_mat(self):
        xp = utils.ProcessingContext().xp
        return xp.dot(xp.asarray(self.intrinsic_mat), self.extrinsic_mat)

    @property
    def is_characterised(self):
        return  (self.intrinsic_mat is not None) \
                and (self.distort_mat is not None) \
                and (self.board_poses is not None)

    @property
    def reprojection_err(self):
        if self._reproj_errs is None: return
        
        xp = utils.ProcessingContext().xp

        return xp.sqrt(xp.mean(self._reproj_errs ** 2))

    def execute(self, board: CharacterisationBoard, poi_coords, resolution, extraFlags=None):
        w, h = resolution

        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp

            board_coords = utils.ToContext(xp, board.get_poi_coords())

            objectCoords = xp.repeat(board_coords[xp.newaxis, ...], len(poi_coords), axis=0)

            poi_coords = utils.ToContext(xp, poi_coords)

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
                objectCoords, poi_coords, (w, h), kGuess, utils.ToContext(xp, self.distort_mat), flags=flags
            )

            self._pose_rotations = xp.asarray(self._pose_rotations)
            self._pose_translations = xp.asarray(self._pose_translations)

            self._reproj_errs = self.calc_reproj_errs(objectCoords, poi_coords,
                self.intrinsic_mat, self.distort_mat, self._pose_rotations, self._pose_translations)
            
            self._reproj_errs = self._reproj_errs.flatten()

            self._rotation = cv2.Rodrigues(self._pose_rotations[0])[0]
            self._translation = self._pose_translations[0].squeeze()

            self._resolution = resolution
            self._pose_poi_coords = poi_coords

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

    def joint_char(self, other: ZhangChar, board: CharacterisationBoard, flags=0):
        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

            board_coords = board.get_poi_coords()
            obj_coords = xp.repeat(board_coords[xp.newaxis, ...], len(self.pose_poi_coords), axis=0)

            joint_rms_reproj, self._intrinsic_mat, self._distort_mat, other._intrinsic_mat, other._distort_mat, self._rotation_to_other, self._translation_to_other, E, F = \
                cv2.stereoCalibrate(obj_coords, self.pose_poi_coords, other.pose_poi_coords,
                self.intrinsic_mat, self.distort_mat, other.intrinsic_mat, other.distort_mat,
                self.resolution, criteria=criteria, flags=flags
            )

            other._rotation_to_other = self.rotation_to_other.T
            other._translation_to_other = -other._rotation_to_other @ self.translation_to_other

            return joint_rms_reproj

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
            reproj_rms = xp.sqrt(xp.mean(xp.power(self._reproj_errs, 2)))
            v += f' Reprojection Error: {reproj_rms:.4f}'

        return v
    

# Interfaces

class ICharable(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_char(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_resolution(self):
        raise NotImplementedError
        