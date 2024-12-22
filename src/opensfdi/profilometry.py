import numpy as np
import pydantic

from typing import Optional
from abc import ABC, abstractmethod
from numpy.polynomial import polynomial as P

# Interfaces / Abstract classes

class BaseProf(pydantic.BaseModel, ABC):
    phasemaps: int = 2
    needs_motor_stage: bool = True

    data: Optional[np.ndarray] = None

    model_config = pydantic.ConfigDict(extra='allow', arbitrary_types_allowed=True)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not issubclass(v, BaseProf):
            raise ValueError("Invalid Object")

        return v

    @abstractmethod
    def calibrate(self, phasemaps, heights):
        if (heights is None): raise TypeError

        # Check passed number of heights equals number of img steps
        if (ph_count := phasemaps.shape[0]) != (h_count := heights.shape[0]):
            raise Exception(f"You must provide an equal number of heights to phasemaps ({ph_count} phasemaps, {h_count} heights given)")

    @abstractmethod
    def heightmap(self, phasemaps):
        # TODO: None check for phasemaps

        # Check passed number of heights equals number of img steps
        if (li := phasemaps.shape[0]) != (lh := self.phasemaps):
            raise Exception(f"Provided number of phase maps is incorrect ({li} passed, {lh} needed)")

    def __str__(self):
        ran = "Not calibrated" if self.data is None else "Calibrated"
        return f"{type(self)} : {ran}"

class ClassicProf(BaseProf):
    """ 
        Classic phase-height model

        Note: Camera and Projector must both be perpendicular to reference plane

        Extend this class and overload the calibrate / heightmap methods for your own functionality

        Args:
            name : Name (id-like) associated with the calibration
            data : Calibration data required for heightmap reconstruction
    """

    def calibrate(self, phasemaps, heights):
        pass

    def heightmap(self, phasemaps):
        """ TODO: Complete documentation """
        super().heightmap(phasemaps)

        ref_phase = phasemaps[0]
        meas_phase = phasemaps[1]

        phase_diff = meas_phase - ref_phase

        # h = 𝜙𝐷𝐸 ⋅ 𝑝 ⋅ 𝑑 / 𝜙𝐷𝐸 ⋅ 𝑝 + 2𝜋𝑙

        a = phase_diff * self.data[0] * self.data[2]
        b = phase_diff * self.data[0] + 2.0 * np.pi * self.data[1]
        
        self.data = a / b
        
        return self.data

class LinearInverseProf(BaseProf):
    def calibrate(self, phasemaps, heights):
        """
            The linear inverse calibration model for fringe projection setups.

            Note:   
                - The moving plane must be parallel to the camera 
                - The first phasemap is taken to be the reference phasemap

                Δ𝜙(x, y) = h(x, y)Δ𝜙(x, y)a(x, y) + h(x, y)b(x, y)
        """

        super().calibrate(phasemaps, heights)

        # Calculate phase difference maps at each height
        # Phase difference between ref and h = 0 is zero
        z, h, w = phasemaps.shape
        ref_phase = phasemaps[0] # Assume reference phasemap is first entry

        ph_maps = np.empty(shape=(z, h, w))
        ph_maps[0] = 0.0
        ph_maps[1:] = phasemaps[1:] - ref_phase

        # Least squares fit on a pixel-by-pixel basis to its height value (a, b)
        self.data = np.empty(shape=(2, h, w), dtype=np.float64)

        # Δ𝜙(x, y) = h(x, y)Δ𝜙(x, y)a(x, y) + h(x, y)b(x, y)

        for y in range(h):
            for x in range(w):
                t = heights * ph_maps[:, y, x]

                A = np.vstack([t, np.ones(len(t))]).T
                m, c = np.linalg.lstsq(A, heights)[0]

                self.data[0, y, x] = m
                self.data[1, y, x] = c

    def heightmap(self, phasemaps):
        """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """

        super().heightmap(phasemaps)

        # Obtain phase difference (reference phasemap should be first, measurement second)
        phase_diff = phasemaps[1] - phasemaps[0]

        # Apply calibrated polynomial values to each pixel of the phase difference
        h, w = phase_diff.shape
        self.data = np.empty_like(phase_diff)

        for y in range(h):
            for x in range(w):
                self.data [y, x] = phase_diff[y, x] / (phase_diff[y, x] * self.data[0, y, x] + self.data[1, y, x])

        return self.data

class PolynomialProf(BaseProf):
    degree: Optional[int] = 5

    def calibrate(self, phasemaps, heights):
        """
            The polynomial calibration model for fringe projection setups.

            Note:   
                - The moving plane must be parallel to the camera 
                - The first phasemap is taken to be the reference phasemap
        """

        super().calibrate(phasemaps, heights)

        # Check polynomial degree is greater than zero
        if self.degree < 1: raise ValueError("Degree of the polynomial must be greater than zero")

        # Calculate phase difference maps at each height
        # Phase difference between ref and h = 0 is zero
        _, h, w = phasemaps.shape

        ph_maps = np.empty_like(phasemaps)
        ph_maps[0] = 0.0                            # Phase difference between baseline and baseline is 0.0
        ph_maps[1:] = phasemaps[1:] - phasemaps[0]  # Phase difference between baseline and height-increments

        # Polynomial fit on a pixel-by-pixel basis to its height value
        self.data = np.empty(shape=(self.degree + 1, h, w), dtype=np.float64)

        for y in range(h):
            for x in range(w):
                self.data[:, y, x] = P.polyfit(ph_maps[:, y, x], heights, deg=self.degree)

        return self.data

    def heightmap(self, phasemaps):
        """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """
        super().heightmap(phasemaps)

        # Obtain phase difference
        ref_phase = phasemaps[0]
        img_phase = phasemaps[1]
        phase_diff = img_phase - ref_phase

        # Apply calibrated polynomial values to each pixel of the phase difference
        h, w = phase_diff.shape
        heightmap = np.zeros_like(phase_diff)

        for y in range(h):
            for x in range(w):
                heightmap[y, x] = P.polyval(phase_diff[y, x], self.data[:, y, x])

        return heightmap

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