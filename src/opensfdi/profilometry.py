import cv2
import numpy as np
import time


from abc import ABC, abstractmethod
from numpy.polynomial import polynomial as P
from matplotlib import pyplot as plt

from . import phase, image, devices

# Reconstruction classes

class ProfilRegistry:
    _registry = {}

    @classmethod
    def register(cls, clazz):
        cls._registry[clazz.__name__] = clazz
        return clazz

    @classmethod
    def get_class(cls, name):
        return cls._registry[name]

class IReconstructor(ABC):
    @abstractmethod
    def __init__(self):
        # Do I need to add metadata here? Not sure
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, camera: devices.Camera, projector: devices.Projector, phi_shift: phase.PhaseShift, phi_unwrap: phase.PhaseUnwrap, num_stripes):
        raise NotImplementedError
    
    @abstractmethod
    def _gather_phasemap(self, camera: devices.Camera, projector: devices.Projector, phi_shift: phase.PhaseShift, phi_unwrap: phase.PhaseUnwrap):
        raise NotImplementedError
    
    def _gather_dc_img(self, camera: devices.Camera, projector: devices.Projector):
        # Gather a single full intensity DC img
        return devices.fringe_project(camera, projector, 0.0, [0.0])[0]

class PhaseHeightReconstructor(IReconstructor):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, camera: devices.Camera, projector: devices.Projector, phi_shift: phase.PhaseShift, phi_unwrap: phase.PhaseUnwrap, num_stripes):
        raise NotImplementedError
    
    def _gather_phasemap(self, camera: devices.Camera, projector: devices.Projector, phi_unwrap: phase.PhaseUnwrap, phi_shift: phase.PhaseShift):
        # TODO: Add some flag to save the images whilst gathering?
        sfs = phi_unwrap.get_fringe_count()
        phases = phi_shift.get_phases()

        shifted = np.empty((len(sfs), *camera.shape), dtype=np.float32)
        
        # Calculate the wrapped phase maps
        for j, sf in enumerate(sfs):
            imgs = devices.fringe_project(camera, projector, sf, phases)
            shifted[j] = phi_shift.shift(imgs)

        # Calculate unwrapped phase maps
        return phi_unwrap.unwrap(shifted)

class StereoReconstructor(IReconstructor):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    def reconstruct(self, camera: devices.Camera, projector: devices.Projector, phi_shift: phase.PhaseShift, phi_unwrap: phase.PhaseUnwrap, num_stripes):
        if not camera.calibrated:
            raise Exception("You must use a calibrated camera for reconstruction")
        
        if not projector.calibrated:
            raise Exception("You must use a calibrated projector for reconstruction")
        
        return None

    def _gather_phasemap(self, camera: devices.Camera, projector: devices.Projector, phi_unwrap: phase.PhaseUnwrap, phi_shift: phase.PhaseShift):
        # TODO: Add some flag to save the images whilst gathering?
        sfs = phi_unwrap.get_fringe_count()
        phases = phi_shift.get_phases()

        shifted = np.empty((len(sfs), *camera.shape), dtype=np.float32)
        
        # Calculate the wrapped phase maps
        for j, sf in enumerate(sfs):
            imgs = devices.fringe_project(camera, projector, sf, phases)
            shifted[j] = phi_shift.shift(imgs)

        # Calculate unwrapped phase maps
        return phi_unwrap.unwrap(shifted)

@ProfilRegistry.register
class PolynomialProfil(PhaseHeightReconstructor):
    def __init__(self, polydata):
        self.polydata = polydata

    def reconstruct(self, camera: devices.Camera, projector: devices.Projector, phi_shift: phase.PhaseShift, phi_unwrap: phase.PhaseUnwrap, num_stripes):
        # """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """

        # raw_imgs = np.array([img.data for img in imgs])
        # grey_imgs = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in raw_imgs])
        
        # # Calculate a mask to use for segmentation
        # dc_img = image.to_int8(image.dc_imgs(grey_imgs))
        # ret, mask = cv2.threshold(dc_img, 16, 255, cv2.THRESH_BINARY)
        # image.show_image(image.RawImage(dc_img))

        # # Apply mask to imgs
        # grey_imgs = [cv2.bitwise_and(img, img, mask=mask) for img in grey_imgs]
        # # contours, hierarchy = cv2.findContours(image=thresh_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # # Make sure to check if horizontal or vertical ! (assumes horizontal by default)
        # if unwrapper.phase_count == 1:
        #     shifted = shifter.shift(grey_imgs)
        # else:
        #     l = shifter.phase_count
        #     shifted = [shifter.shift(grey_imgs[i * l::l]) for i in range(unwrapper.phase_count)]

        # phasemap = unwrapper.unwrap(shifted)
        # phase.show_phasemap(phasemap)    

        print(self.polydata.shape)

        # Obtain phase difference
        phase_diff = meas_phasemap - ref_phasemap

        # Apply calibrated polynomial values to each pixel of the phase difference
        h, w = phase_diff.shape
        heightmap = np.zeros_like(phase_diff)

        for y in range(h):
            for x in range(w):
                heightmap[y, x] = P.polyval(phase_diff[y, x], self.polydata[:, y, x])

        return heightmap

@ProfilRegistry.register
class StereoProfil(StereoReconstructor):
    def __init__(self):
        pass

    def _points1(self, P_c, P_p, x_c, y_c, x_p):
        a1 = P_c[0, 0] - x_c * P_c[2, 0]
        a2 = P_c[0, 1] - x_c * P_c[2, 1]
        a3 = P_c[0, 2] - x_c * P_c[2, 2]
        a4 = P_c[1, 0] - y_c * P_c[2, 0]
        a5 = P_c[1, 1] - y_c * P_c[2, 1]
        a6 = P_c[1, 2] - y_c * P_c[2, 2]
        a7 = P_p[0, 0] - x_p * P_p[2, 0]
        a8 = P_p[0, 1] - x_p * P_p[2, 1]
        a9 = P_p[0, 2] - x_p * P_p[2, 2]

        b1 = x_c * P_c[2, 3] - P_c[0, 3]
        b2 = y_c * P_c[2, 3] - P_c[1, 3]
        b3 = x_p * P_p[2, 3] - P_p[0, 3]

        D = -a3 * a5 * a7 + a2 * a6 * a7 + a3 * a4 * a8 - a1 * a6 * a8 - a2 * a4 * a9 + a1 * a5 * a9
        x_w = (1. / D) * ((a5 * a9 - a6 * a8) * b1 + (a3 * a8 - a2 * a9) * b2 + (a2 * a6 - a3 * a5) * b3)
        y_w = (1. / D) * ((a6 * a7 - a4 * a9) * b1 + (a1 * a9 - a3 * a7) * b2 + (a3 * a4 - a1 * a6) * b3)
        z_w = (1. / D) * ((a4 * a8 - a5 * a7) * b1 + (a2 * a7 - a1 * a8) * b2 + (a1 * a5 - a2 * a4) * b3)

        points = np.dstack([x_w, y_w, z_w])

        return points.reshape(-1, 3)

    def _points2(self, P_c, P_p, x_c, y_c, x_p):
        c_height, c_width = x_c.shape

        A = np.empty((c_height, c_width, 3, 3))
        b = np.empty((c_height, c_width, 3, 1))

        A[:, :, 0, 0] = P_c[0, 0] - P_c[2, 0] * x_c
        A[:, :, 0, 1] = P_c[0, 1] - P_c[2, 1] * x_c
        A[:, :, 0, 2] = P_c[0, 2] - P_c[2, 2] * x_c

        A[:, :, 1, 0] = P_c[1, 0] - P_c[2, 0] * y_c
        A[:, :, 1, 1] = P_c[1, 1] - P_c[2, 1] * y_c
        A[:, :, 1, 2] = P_c[1, 2] - P_c[2, 2] * y_c

        A[:, :, 2, 0] = P_p[0, 0] - P_p[2, 0] * x_p
        A[:, :, 2, 1] = P_p[0, 1] - P_p[2, 1] * x_p
        A[:, :, 2, 2] = P_p[0, 2] - P_p[2, 2] * x_p

        b[:, :, 0, 0] = P_c[0, 3] - P_c[2, 3] * x_c
        b[:, :, 1, 0] = P_c[1, 3] - P_c[2, 3] * y_c
        b[:, :, 2, 0] = P_p[0, 3] - P_p[2, 3] * x_p

        pc = -np.linalg.solve(A, b)

        return pc.reshape(-1, 3)

    def reconstruct(self, camera: devices.Camera, projector: devices.Projector, phi_shift: phase.PhaseShift, phi_unwrap: phase.PhaseUnwrap, num_stripes):
        """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """

        super().reconstruct(camera, projector, phi_unwrap, phi_shift, num_stripes)

        # Gather a phasemap by using the camera and projector
        phasemap = self._gather_phasemap(camera, projector, phi_unwrap, phi_shift)

        c_height, c_width = camera.resolution
        p_height, p_width = projector.resolution

        x_c, y_c = np.meshgrid(np.arange(c_width, dtype=np.float32), np.arange(c_height, dtype=np.float32))
        x_p = (phasemap * p_height) / (2.0 * np.pi * num_stripes)

        # pc = self._points2(camera.proj_mat, projector.proj_mat, x_c, y_c, x_p)
        pc = self._points1(camera.proj_mat, projector.proj_mat, x_c, y_c, x_p)

        # Find the indices of any NaNs
        valid_points = ~np.isnan(pc).any(axis=1)

        # Remove any NaNs for masked values earlier
        return pc[valid_points], valid_points


# Calibration classes

class FPCalibrator(ABC):
    @abstractmethod
    def __init__(self):
        # TODO: Allow for other calibration artefacts
        pass

class StereoCalibrator(FPCalibrator):
    def __init__(self, calib_board: devices.CalibrationBoard):
        super().__init__()

        self.__calib_board = calib_board

    def gather_phasemap(self, camera: devices.Camera, projector: devices.Projector, phi_shift: phase.PhaseShift, phi_unwrap: phase.PhaseUnwrap):
        # TODO: Add some flag to save the images whilst gathering?
        sfs = phi_unwrap.get_fringe_count()
        phases = phi_shift.get_phases()

        shifted = np.empty((len(sfs), *camera.shape), dtype=np.float32)
        
        dc_img = None

        # Calculate the wrapped phase maps
        for j, sf in enumerate(sfs):
            imgs = devices.fringe_project(camera, projector, sf, phases)

            if j == 0: dc_img = image.dc_imgs(imgs)

            shifted[j] = phi_shift.shift(imgs)

        # Calculate unwrapped phase maps
        return phi_unwrap.unwrap(shifted), dc_img
    
    def gather_dc_img(self, camera: devices.Camera, projector: devices.Projector):
        # Gather a single full intensity DC img
        return devices.fringe_project(camera, projector, 0.0, [0.0])[0]

    def calibrate(self, camera: devices.Camera, projector: devices.Projector, phi_shift: phase.PhaseShift, phi_unwrap: phase.PhaseUnwrap, num_imgs=15):
        start_time = time.time()

        phasemaps = np.empty(shape=(num_imgs * 2, *camera.shape), dtype=np.float32) 
        dc_imgs = np.empty(shape=(num_imgs, *camera.shape), dtype=np.float32)

        for i in range(num_imgs):
            # Vertical phase maps
            projector.rotation = False
            phasemaps[2*i], dc_imgs[i] = self.gather_phasemap(camera, projector, phi_shift, phi_unwrap)
            # phase.show_phasemap(phasemaps[2*i])
            # image.show_image(dc_imgs[i])
            
            # Horizontal phase maps (ignore DC image as we don't need it)
            projector.rotation = True
            phasemaps[2*i+1], _ = self.gather_phasemap(camera, projector, phi_shift, phi_unwrap)
            # phase.show_phasemap(phasemaps[2*i+1])

        # Calibrate the camera and projector
        self.__calibrate(camera, projector, dc_imgs, phasemaps, num_stripes=phi_unwrap.get_fringe_count()[-1])

        end_time = (time.time() - start_time) * 1000

        result = CalibrationResult()
        result.time_taken = end_time
        result.phi_shifter = phi_shift.__class__.__name__
        result.phi_unwrapper = phi_unwrap.__class__.__name__

        return result

    def __calibrate(self, camera: devices.Camera, projector: devices.Projector, cb_imgs, phasemaps: np.ndarray, num_stripes) -> StereoProfil:
        """ The triangular stereo calibration model for fringe projection setups. """

        N = len(cb_imgs)

        # Corner finding algorithm needs greyscale images
        assert N * 2 == len(phasemaps)

        world_xyz = []
        corner_pixels = []
        valid_phasemaps = []

        corner_coords = self.__calib_board.get_poi_coords()

        for i in range(N): # We can skip duplicate checkerboards as is bad practice
            corners = self.__calib_board.find_pois(cb_imgs[i])

            if corners is None:
                print(f"Could not find checkerboard corners for image {i}")
                continue

            world_xyz.append(corner_coords)
            corner_pixels.append(corners)

            # Add the phasemaps
            valid_phasemaps.append(phasemaps[2*i])
            valid_phasemaps.append(phasemaps[2*i+1])

            # DEBUG: Draw corners detected
            # image.show_image(cv2.drawChessboardCorners(cb_imgs[i], self.__cb_size, corners, True))

        world_xyz = np.array(world_xyz)
        corner_pixels = np.array(corner_pixels)

        # Finished phase manipulation, now just need to calibrate camera and projector
        print(f"{len(corner_pixels)} images with POIs correctly identified")

        E_c = camera.calibrate(world_xyz, corner_pixels)
        E_p = projector.calibrate(world_xyz, corner_pixels, valid_phasemaps, num_stripes)

        print(f"Camera reprojection error: {E_c}")
        print(f"Projector reprojection error: {E_p}")

        E_total = self.refine_devices(camera, projector, world_xyz, corner_pixels)
        print(f"Total reprojection error: {E_total}")

        self._metadata = {
            "images_used"       : N,
            "cbs_detected"      : len(world_xyz),

            "time_taken"        : 0.0,
            "cam_reproj_err"    : E_c,
            "proj_reproj_err"   : E_p,
        }

        return E_c, E_p

    def refine_devices(self, camera: devices.Camera, projector: devices.Projector, world_xyz, corner_pixels):
        # calibrate stereo
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC

        reproj, camera.K, camera.dist_mat, projector.K, projector.dist_mat, R, T, E, F = cv2.stereoCalibrate(world_xyz, 
            corner_pixels, projector.proj_coords,
            camera.K, camera.dist_mat,
            projector.K, projector.dist_mat,
            None, flags=flags)
        
        # Update projector rotation and translation
        projector.R = np.dot(R, camera.R)
        projector.t = np.dot(R, camera.t) + T
        
        return reproj

class IPhaseHeightCalibrator(ABC):
    @abstractmethod
    def calibrate(self, phasemaps, heights) -> PhaseHeightReconstructor:
        raise NotImplementedError

class PolynomialProfilCalibrator(IPhaseHeightCalibrator):
    def __init__(self, degree=5):
        self.degree = degree

    def calibrate(self, phasemaps, heights) -> PolynomialProfil:
        """
            The polynomial calibration model for fringe projection setups.

            Note:   
                - The moving plane must be parallel to the camera 
                - The first phasemap is taken to be the reference phasemap
        """
        # Calculate phase difference maps at each height
        # Phase difference between ref and h = 0 is zero
        _, h, w = phasemaps.shape

        ph_maps = np.empty_like(phasemaps)
        ph_maps[0] = 0.0                            # Phase difference between baseline and baseline is 0.0
        ph_maps[1:] = phasemaps[1:] - phasemaps[0]  # Phase difference between baseline and height-increments

        # Polynomial fit on a pixel-by-pixel basis to its height value
        polydata = np.empty(shape=(self.degree + 1, h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                polydata[:, y, x] = P.polyfit(ph_maps[:, y, x], heights, deg=self.degree)

        return PolynomialProfil(polydata)

class CalibrationResult:
    def __init__(self):
        self.time_taken = None
        self.phi_unwrapper = None
        self.phi_shifter = None

        self.number_of_calibration_images = None

class MeasurementResult:
    def __init__(self):
        self.time_taken = None

# Utility functions

def checkerboard_centre(cb_size, square_size):
    cx = (cb_size[0] / 2 - 1) * square_size
    cy = (cb_size[1] / 2 - 1) * square_size 

    return cx, cy

# class TriangularStereoHeight(PhaseHeight):
#     def to_stl(self, heightmap):
#         # Create vertices from the heightmap
#         vertices = []
#         for y in range(heightmap.shape[0]):
#             for x in range(heightmap.shape[1]):
#                 vertices.append([x, y, heightmap[y, x]])

#         vertices = np.array(vertices)

#         # Create faces for the mesh
#         faces = []
#         for y in range(heightmap.shape[0] - 1):
#             for x in range(heightmap.shape[1] - 1):
#                 v1 = x + y * heightmap.shape[1]
#                 v2 = (x + 1) + y * heightmap.shape[1]
#                 v3 = x + (y + 1) * heightmap.shape[1]
#                 v4 = (x + 1) + (y + 1) * heightmap.shape[1]

#                 # First triangle
#                 faces.append([v1, v2, v3])
#                 # Second triangle
#                 faces.append([v2, v4, v3])

#         # Create the mesh object
#         # mesh_data = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
#         # for i, f in enumerate(faces):
#         #     for j in range(3):
#         #         mesh_data.vectors[i][j] = vertices[f[j]]

#         # mesh_data.save('heightmap_mesh.stl')