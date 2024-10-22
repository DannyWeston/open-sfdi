import numpy as np

from abc import ABC, abstractmethod
from numpy.polynomial import polynomial as P

class PhaseHeight(ABC):
    @abstractmethod
    def __init__(self, calib=None):
        self._calib = calib

        self.__phasemaps_needed = 2

        self.__post_cbs = []

    @abstractmethod
    def calibrate(self, phasemaps, *args, **kwargs):
        if (phasemaps is None): raise TypeError
        
    @abstractmethod
    def heightmap(self, phasemaps, *args, **kwargs):
        if self._calib is None:
            raise Exception("You need to run/load calibration data first")
        
        if phasemaps is None: raise TypeError

    @property
    def phasemaps_needed(self):
        return self.__phasemaps_needed
    
    @property
    def post_cbs(self):
        return self.__post_cbs
        
    def add_post_ref_cb(self, cb):
        """ TODO: Add description """
        self.__post_cbs.append(cb)

    def call_post_cbs(self):
        for cb in self.__post_cbs: cb()

    def save_data(self, path):
        if self._calib is None:
            raise Exception("You need to run/load calibration data first")

        with open(path, "wb") as out_file:
            np.save(out_file, self._calib)

class PolynomialPH(PhaseHeight):
    def __init__(self, calib=None):
        super().__init__(calib)

        if not (self._calib is None):
            h, w, cs = self._calib.shape
            self.__degree = cs

    @property
    def degree(self):
        """ The degree of the polynomial used for the last calibration """
        return self.__degree

    def calibrate(self, phasemaps, heights, degree):
        """ 
            The polynomial calibration model for fringe projection setups.

            Note:   
                - The moving plane must be parallel to the camera 
                - The first img value is taken to be the reference
        """
        super().calibrate(phasemaps)

        if (heights is None): raise TypeError

        # Check polynomial degree is greater than zero
        if degree < 1: raise ValueError("Degree of the polynomial must be greater than zero")
        self.__degree = degree

        # Check passed number of heights equals numebr of img steps
        if (li := len(phasemaps)) != (lh := len(heights)): 
            raise ValueError(f"You must provide an equal number of heights to phasemaps ({li} and {lh} given)")

        # Calculate phase difference maps at each height
        # Phase difference between ref and h = 0 is zero
        z, h, w = phasemaps.shape
        ref_phase = phasemaps[0] # Assume reference phasemap is first entry

        ph_maps = np.empty(shape=(z, h, w))
        ph_maps[0] = 0.0
        ph_maps[1:] = phasemaps[1:] - ref_phase

        # Polynomial fit on a pixel-by-pixel basis to its height value
        self._calib = np.empty(shape=(degree + 1, h, w), dtype=np.float64)

        for y in range(h):
            for x in range(w):
                self._calib[:, y, x] = P.polyfit(ph_maps[:, y, x], heights, deg=degree)

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
                heightmap[y, x] = P.polyval(phase_diff[y, x], self._calib[:, y, x])

        return heightmap

# def linear_inverse_phase_height(imgs, dists):
#     # Calculate least squares for each pixel
#     ph_maps = np.array([unwrapped_phase(wrapped_phase(phases)) for phases in imgs])

#     # Reshape for pixel-wise computation
#     n, h, w = ph_maps.shape

#     coeffs = np.empty(shape=(h, w, 2), dtype=np.float64)

#     for i, h in enumerate(ph_maps.reshape(h, w, n)):
#         for j, pixel_vals in enumerate(h):
#             X = np.column_stack((dists * pixel_vals, dists))

#             coeffs[i, j] = np.linalg.lstsq(X, pixel_vals, rcond=None)[0]

#     return coeffs

# def linear_inverse_phase_height_2(imgs, dists):
#     # 1 / ℎ(𝑥, 𝑦) = 𝑎(𝑥, 𝑦) + 𝑏(𝑥, 𝑦) / Δ𝜙(𝑥, 𝑦)
#     phasemaps = np.array([unwrapped_phase(wrapped_phase(phases)) for phases in imgs])
    
#     # Calculate inverses
#     delta_p_ = 1 / phasemaps
#     h_ = 1 / dists

#     # Calculate least squares for each pixel
#     n, h, w = delta_p_.shape
#     delta_p_ = delta_p_.reshape(h, w, n) # Reshape for pixel-wise computation

#     coeffs = np.empty(shape=(h, w, 2), dtype=np.float64)

#     for i, h in enumerate(delta_p_):
#         for j, pixels in enumerate(h):
#             A = np.vstack([np.ones(len(pixels)), pixels]).T

#             coeffs[i, j] = np.array(np.linalg.lstsq(A, h_)[0])

#     return coeffs

# class PhaseHeight(ABC):
#     def phasemap(self, imgs):
#         w_phase = wrapped_phase(imgs)
        
#         return unwrapped_phase(w_phase)
    
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

# class ClassicPhaseHeight(PhaseHeight):
#     # ℎ = 𝜙𝐷𝐸 ⋅ 𝑝 ⋅ 𝑑 / 𝜙𝐷𝐸 ⋅ 𝑝 + 2𝜋𝑙
#     # p = stripe width
#     # d = distance between camera and reference plane
#     # l = distance between camera and projector
        
#     def __init__(self, p: float, d: float, l: float):
#         """ p = Stripe width,
#             d = Distance between camera and reference plane,
#             l = distance between camera and projector """
#         super().__init__()
        
#         self.p = p
#         self.d = d 
#         self.l = l
    
#     def heightmap(self, ref_imgs, imgs, convert_grey=False, crop=None):
#         if convert_grey:
#             imgs = np.array([rgb2grey(img) for img in imgs])
#             ref_imgs = np.array([rgb2grey(img) for img in ref_imgs])
  
#         if crop is not None:
#             h, w = imgs[0].shape[:2]
#             if len(crop) == 2:
#                 crop_x1 = int(crop[0] * w)
#                 crop_x2 = w - crop_x1 - 1
#                 crop_y1 = int(crop[1] * h)
#                 crop_y2 = h - crop_y1 - 1
#             elif len(crop) == 4:
#                 crop_x1 = int(crop[0] * w)
#                 crop_y1 = int(crop[1] * h)
#                 crop_x2 = w - int(crop[2] * w) - 1
#                 crop_y2 = h - int(crop[3] * h) - 1
#             else: raise Exception("Invalid crop tuple passed")
            
#             imgs = np.array([centre_crop_img(img, crop_x1, crop_y1, crop_x2, crop_y2) for img in imgs])
#             ref_imgs = np.array([centre_crop_img(img, crop_x1, crop_y1, crop_x2, crop_y2) for img in ref_imgs])
        
#         ref_phase, measured_phase = self.phasemap(ref_imgs), self.phasemap(imgs)

#         phase_diff = measured_phase - ref_phase
        
#         return np.divide(self.l * phase_diff, phase_diff - (2.0 * np.pi * self.p * self.d), dtype=np.float32)

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