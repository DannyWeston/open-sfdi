import numpy as np
import open3d as o3d
import cv2

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
        pass

    @abstractmethod
    def reconstuct(self, phasemaps, **kwargs):
        raise NotImplementedError

class IPhaseHeightReconstructor(IReconstructor):
    @abstractmethod
    def __init__(self, camera: devices.Camera, projector: devices.Projector):
        if not camera.is_calibrated():
            raise Exception("You must use a calibrated camera for the stereo method")
        
        if not projector.is_calibrated():
            raise Exception("You must use a calibrated projector for the stereo method")

        self.camera = camera
        self.projector = projector

class IStereoReconstructor(IReconstructor):
    @abstractmethod
    def __init__(self, camera: devices.Camera, projector: devices.Projector, vertical: bool):
        if not camera.is_calibrated():
            raise Exception("You must use a calibrated camera for the stereo method")
        
        if not projector.is_calibrated():
            raise Exception("You must use a calibrated projector for the stereo method")

        self.camera = camera
        self.projector = projector

        self._vertical = vertical

    @abstractmethod
    def reconstruct(self, phasemap, stripe_pixels):
        raise NotImplementedError

    def get_rotation(self):
        return 0.0 if self._vertical else np.pi / 2.0

@ProfilRegistry.register
class PolynomialProfil(IPhaseHeightReconstructor):
    def __init__(self, polydata):
        self.polydata = polydata

    def reconstruct(self, ref_phasemap, meas_phasemap):
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
class StereoProfil(IStereoReconstructor):
    def __init__(self, camera: devices.Camera, projector: devices.Projector, vertical=True):
        super().__init__(camera, projector, vertical)

    def reconstruct(self, phasemap, num_stripes: float):
        """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """

        # # Make sure to check if horizontal or vertical ! (assumes horizontal by default)
        # if unwrapper.phase_count == 1:
        #     shifted = shifter.shift(grey_imgs)
        # else:
        #     l = shifter.phase_count
        #     shifted = [shifter.shift(grey_imgs[i * l::l]) for i in range(unwrapper.phase_count)]

        c_height, c_width = self.camera.resolution
        stripe_pixels = (c_width if self._vertical else c_height) / num_stripes

        x_c, y_c = np.meshgrid(np.arange(c_width, dtype=np.float32), np.arange(c_height, dtype=np.float32))

        x_p = (phasemap * stripe_pixels) / (2.0 * np.pi)

        # Extract x_w, y_w, z_w from the solution
        # Matrix equations are as following:
        # P_c(0, 0) - u_c * P_c(2, 0)     P_c(0, 1) - u_c * P_c(2, 1)   P_c(0, 2) - u_c * P_c(2, 2)
        # P_c(1, 0) - v_c * P_c(2, 0)     P_c(1, 1) - v_c * P_c(2, 1)   P_c(1, 2) - v_c * P_c(2, 2)
        # P_p(0, 0) - u_p * P_p(2, 0)     P_p(0, 1) - u_p * P_p(2, 1)   P_p(0, 2) - u_p * P_p(2, 2)

        P_c = self.camera.proj_mat
        P_p = self.projector.proj_mat

        A = np.empty((c_height, c_width, 3, 3))
        b = np.empty((c_height, c_width, 3, 1))

        # Method 1
        # A[:, :, 0, 0] = P_c[0, 0] - (u_c * P_c[2, 0])
        # A[:, :, 0, 1] = P_c[0, 1] - (u_c * P_c[2, 1])
        # A[:, :, 0, 2] = P_c[0, 2] - (u_c * P_c[2, 2])

        # A[:, :, 1, 0] = P_c[1, 0] - (v_c * P_c[2, 0])
        # A[:, :, 1, 1] = P_c[1, 1] - (v_c * P_c[2, 1])
        # A[:, :, 1, 2] = P_c[1, 2] - (v_c * P_c[2, 2])

        # A[:, :, 2, 0] = P_p[1, 0] - (u_p * P_p[2, 0])
        # A[:, :, 2, 1] = P_p[1, 1] - (u_p * P_p[2, 1])
        # A[:, :, 2, 2] = P_c[1, 2] - (u_p * P_c[2, 2])

        # b[:, :, 0, 0] = P_c[0, 3] - (u_c * P_c[2, 3])
        # b[:, :, 1, 0] = P_c[1, 3] - (v_c * P_c[2, 3])
        # b[:, :, 2, 0] = P_c[1, 3] - (u_p * P_c[2, 3])

        # Method 2
        A[:, :, 0, 0] = P_c[0, 0] - (x_c * P_c[2, 0])
        A[:, :, 0, 1] = P_c[0, 1] - (x_c * P_c[2, 1])
        A[:, :, 0, 2] = P_c[0, 2] - (x_c * P_c[2, 2])

        A[:, :, 1, 0] = P_c[1, 0] - (y_c * P_c[2, 0])
        A[:, :, 1, 1] = P_c[1, 1] - (y_c * P_c[2, 1])
        A[:, :, 1, 2] = P_c[1, 2] - (y_c * P_c[2, 2])

        A[:, :, 2, 0] = P_p[0, 0] - (x_p * P_p[2, 0])
        A[:, :, 2, 1] = P_p[0, 1] - (x_p * P_p[2, 1])
        A[:, :, 2, 2] = P_p[0, 2] - (x_p * P_p[2, 2])

        b[:, :, 0, 0] = (x_c * P_c[2, 3]) - P_c[0, 3]
        b[:, :, 1, 0] = (y_c * P_c[2, 3]) - P_c[1, 3]
        b[:, :, 2, 0] = (x_p * P_p[2, 3]) - P_p[0, 3]

        return np.squeeze(np.linalg.solve(A, b))

# Calibration classes

class FPCalibrator(ABC):
    @abstractmethod
    def __init__(self, camera: devices.Camera, projector: devices.Projector, phi_shifter: phase.PhaseShift, phi_unwrapper: phase.PhaseUnwrap):
        self.camera = camera
        self.projector = projector

        self._window_size = (15, 15)
        
        # TODO: Allow for other calibration artefacts
        self._cb_size = (7, 5)

        self._phi_shifter = phi_shifter
        self._phi_unwrapper = phi_unwrapper

class StereoCalibrator(FPCalibrator):
    def __init__(self, phi_shifter: phase.PhaseShift, phi_unwrapper: phase.PhaseUnwrap):
        self.__window_size = (16, 16)
        self.__cb_size = (7, 5)

        self._phi_shifter = phi_shifter
        self._phi_unwrapper = phi_unwrapper

    def gather(self, camera: devices.Camera, projector: devices.Projector) -> tuple:
        # TODO: Add some flag to save the images whilst gathering?
        sfs = self._phi_unwrapper.get_fringe_count()
        phases = self._phi_shifter.get_phases()

        shifted = np.empty((len(sfs), *camera.shape), dtype=np.float32)

        # Calculate the wrapped phase maps
        for j, sf in enumerate(sfs):

            imgs = devices.fringe_project(camera, projector, sf, phases)
            shifted[j] = self._phi_shifter.shift(imgs)

        # Calculate DC Images
        dc_img = image.dc_imgs(imgs)

        # Calculate unwrapped phase maps
        phasemap = self._phi_unwrapper.unwrap(shifted, projector.rotation)

        return dc_img, phasemap

    def calibrate(self, camera: devices.Camera, projector: devices.Projector, num_imgs=15):
        phasemaps = np.empty(shape=(num_imgs, *camera.shape), dtype=np.float32)
        dc_imgs = np.empty_like(phasemaps)

        fringe_count = self._phi_unwrapper.get_fringe_count()

        for i in range(0, num_imgs, 2):
            projector.rotation = True
            dc_imgs[i], phasemaps[i] = self.gather(camera, projector)

            projector.rotation = False
            dc_imgs[i+1], phasemaps[i+1] = self.gather(camera, projector)
        
        return self.__calibrate(camera, projector, dc_imgs, phasemaps, fringe_count[-1])

    def __calibrate(self, camera: devices.Camera, projector: devices.Projector, cb_imgs, phasemaps: np.ndarray, num_stripes) -> StereoProfil:
        """ The triangular stereo calibration model for fringe projection setups. """

        # Corner finding algorithm needs greyscale images
        assert len(cb_imgs) == len(phasemaps)

        world_xyz = []
        corner_pixels = []
        valid_phasemaps = []

        h, w = self.__cb_size
        cb_corners = np.zeros((h * w, 3), np.float32)
        cb_corners[:, :2] = np.mgrid[:h, :w].T.reshape(-1, 2)

        for i in range(0, len(cb_imgs), 2): # We can skip duplicate checkerboards as is bad practice
            corners = image.find_corners(cb_imgs[i], self.__cb_size, self.__window_size)
            corners2 = image.find_corners(cb_imgs[i+1], self.__cb_size, self.__window_size)

            if corners is None:
                print(f"Could not find checkerboard corners for image {i}")
                continue

            if corners2 is None:
                print(f"Could not find checkerboard corners for image {i+1}")
                continue

            # DEBUG: Draw corners detected
            # image.show_image(image.RawImage(cv2.drawChessboardCorners(cb_imgs[i], cb_size, corners, True)))

            world_xyz.append(cb_corners)
            corner_pixels.append(corners)
            valid_phasemaps.append(phasemaps[i])

            world_xyz.append(cb_corners)
            corner_pixels.append(corners2)
            valid_phasemaps.append(phasemaps[i+1])

        # Finished phase manipulation, now just need to calibrate camera and projector
        print(f"{len(valid_phasemaps)} total checkerboards identified")

        E_c = camera.calibrate(world_xyz[::2], corner_pixels[::2]) # Can ignore repeated checkerboards for the camera case
        print(f"Camera reprojection error: {E_c}")

        E_p = projector.calibrate(world_xyz, corner_pixels, valid_phasemaps, num_stripes)        
        print(f"Projector reprojection error: {E_p}")

        return StereoProfil(camera, projector)

class IPhaseHeightCalibrator(ABC):
    @abstractmethod
    def calibrate(self, phasemaps, heights) -> IPhaseHeightReconstructor:
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

# Utility functions

def show_surface(data):
    hf = plt.figure()

    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))

    ha.plot_surface(X, Y, data)

    plt.show()

def show_pointcloud(cloud: np.ndarray, colours=None, name='Point Cloud'):
    # TODO: Something with title
    if cloud.ndim != 3:
        raise Exception("Need to pass a 3D array (width, height, xyz)")

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3))

    if not (colours is None):
        pc.colors = o3d.utility.Vector3dVector(colours.reshape(-1, 3))

    o3d.visualization.draw_geometries([pc])

    o3d.io.write_point_cloud(f"{name}.ply", pc)

# class ProfEnum(Enum):
#     PhaseHeight = 1
#     LinearInverse = 2
#     Polynomial = 3

#     def as_prof_type(self):
#         if self == ProfEnum.PhaseHeight:
#             return PhaseHeight

#         if self == ProfEnum.Polynomial: 
#             return PolynomialPH
    
#         if self == ProfEnum.LinearInverse:
#             return LinearInversePH
        
#         raise Exception("Could not identify profilometry method from enum")


# class TriangularStereoHeight(PhaseHeight):
#     def __init__(self, ref_dist, sensor_dist, freq):
#         super().__init__()
        
#         self.ref_dist = ref_dist
#         self.sensor_dist = sensor_dist
#         self.freq = freq
    
#     def heightmap(self, imgs):
#         phase = self.phasemap(imgs)

#         #heightmap = np.divide(self.ref_dist * phase_diff, 2.0 * np.pi * self.sensor_dist * self.freq)
        
#         #heightmap[heightmap <= 0] = 0 # Remove negative values

#         return None

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