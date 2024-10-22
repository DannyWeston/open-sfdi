import numpy as np

from abc import ABC
from matplotlib import pyplot as plt
from numpy.polynomial import polynomial as P

from skimage.restoration import unwrap_phase

from opensfdi import wrapped_phase, centre_crop_img, rgb2grey, show_phasemap

def show_heightmap(heightmap, title='Heightmap'):
    x, y = np.meshgrid(range(heightmap.shape[0]), range(heightmap.shape[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, np.transpose(heightmap))
    plt.title(title)
    plt.show()

def poly_phase_height(imgs, dists, degree):
    """ The polynomial calibration model for fringe projection setups
        Note: The moving plane must be parallel to the camera 
    """
    z, _, h, w = imgs.shape

    # Calculate phasemap for reference (first img)
    ref_phase = wrapped_phase(imgs[0])  # Obtain wrapped phase
    ref_phase = unwrap_phase(ref_phase) # Obtain unwrapped phase


    # Calculate phase difference maps at each height
    # Phase difference between ref and h = 0 is zero
    ph_maps = np.empty(shape=(z, h, w))
    ph_maps[0] = 0.0

    for i in range(1, z):
        ph_maps[i] = wrapped_phase(imgs[i])     # Obtain wrapped phase
        ph_maps[i] = unwrap_phase(ph_maps[i])   # Obtain unwrapped phase
        ph_maps[i] -= ref_phase                 # Phase difference map (reference phase)

    # Polynomial fit on a pixel-by-pixel basis to its height value
    coeffs = np.empty(shape=(degree + 1, h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            coeffs[:, y, x] = P.polyfit(ph_maps[:, y, x], dists, deg=degree)

    return coeffs

def linear_inverse_phase_height(imgs, dists):
    # Calculate least squares for each pixel
    ph_maps = np.array([unwrapped_phase(wrapped_phase(phases)) for phases in imgs])

    # Reshape for pixel-wise computation
    n, h, w = ph_maps.shape

    coeffs = np.empty(shape=(h, w, 2), dtype=np.float64)

    for i, h in enumerate(ph_maps.reshape(h, w, n)):
        for j, pixel_vals in enumerate(h):
            X = np.column_stack((dists * pixel_vals, dists))

            coeffs[i, j] = np.linalg.lstsq(X, pixel_vals, rcond=None)[0]

    return coeffs

def linear_inverse_phase_height_2(imgs, dists):
    # 1 / ℎ(𝑥, 𝑦) = 𝑎(𝑥, 𝑦) + 𝑏(𝑥, 𝑦) / Δ𝜙(𝑥, 𝑦)
    phasemaps = np.array([unwrapped_phase(wrapped_phase(phases)) for phases in imgs])
    
    # Calculate inverses
    delta_p_ = 1 / phasemaps
    h_ = 1 / dists

    # Calculate least squares for each pixel
    n, h, w = delta_p_.shape
    delta_p_ = delta_p_.reshape(h, w, n) # Reshape for pixel-wise computation

    coeffs = np.empty(shape=(h, w, 2), dtype=np.float64)

    for i, h in enumerate(delta_p_):
        for j, pixels in enumerate(h):
            A = np.vstack([np.ones(len(pixels)), pixels]).T

            coeffs[i, j] = np.array(np.linalg.lstsq(A, h_)[0])

    return coeffs

class PhaseHeight(ABC):
    def phasemap(self, imgs):
        w_phase = wrapped_phase(imgs)
        
        return unwrapped_phase(w_phase)
    
    def to_stl(self, heightmap):
        # Create vertices from the heightmap
        vertices = []
        for y in range(heightmap.shape[0]):
            for x in range(heightmap.shape[1]):
                vertices.append([x, y, heightmap[y, x]])

        vertices = np.array(vertices)

        # Create faces for the mesh
        faces = []
        for y in range(heightmap.shape[0] - 1):
            for x in range(heightmap.shape[1] - 1):
                v1 = x + y * heightmap.shape[1]
                v2 = (x + 1) + y * heightmap.shape[1]
                v3 = x + (y + 1) * heightmap.shape[1]
                v4 = (x + 1) + (y + 1) * heightmap.shape[1]

                # First triangle
                faces.append([v1, v2, v3])
                # Second triangle
                faces.append([v2, v4, v3])

        # Create the mesh object
        # mesh_data = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        # for i, f in enumerate(faces):
        #     for j in range(3):
        #         mesh_data.vectors[i][j] = vertices[f[j]]

        # mesh_data.save('heightmap_mesh.stl')

class ClassicPhaseHeight(PhaseHeight):
    # ℎ = 𝜙𝐷𝐸 ⋅ 𝑝 ⋅ 𝑑 / 𝜙𝐷𝐸 ⋅ 𝑝 + 2𝜋𝑙
    # p = stripe width
    # d = distance between camera and reference plane
    # l = distance between camera and projector
        
    def __init__(self, p: float, d: float, l: float):
        """ p = Stripe width,
            d = Distance between camera and reference plane,
            l = distance between camera and projector """
        super().__init__()
        
        self.p = p
        self.d = d 
        self.l = l
    
    def heightmap(self, ref_imgs, imgs, convert_grey=False, crop=None):
        if convert_grey:
            imgs = np.array([rgb2grey(img) for img in imgs])
            ref_imgs = np.array([rgb2grey(img) for img in ref_imgs])
  
        if crop is not None:
            h, w = imgs[0].shape[:2]
            if len(crop) == 2:
                crop_x1 = int(crop[0] * w)
                crop_x2 = w - crop_x1 - 1
                crop_y1 = int(crop[1] * h)
                crop_y2 = h - crop_y1 - 1
            elif len(crop) == 4:
                crop_x1 = int(crop[0] * w)
                crop_y1 = int(crop[1] * h)
                crop_x2 = w - int(crop[2] * w) - 1
                crop_y2 = h - int(crop[3] * h) - 1
            else: raise Exception("Invalid crop tuple passed")
            
            imgs = np.array([centre_crop_img(img, crop_x1, crop_y1, crop_x2, crop_y2) for img in imgs])
            ref_imgs = np.array([centre_crop_img(img, crop_x1, crop_y1, crop_x2, crop_y2) for img in ref_imgs])
        
        ref_phase, measured_phase = self.phasemap(ref_imgs), self.phasemap(imgs)

        phase_diff = measured_phase - ref_phase
        
        return np.divide(self.l * phase_diff, phase_diff - (2.0 * np.pi * self.p * self.d), dtype=np.float32)

class TriangularStereoHeight(PhaseHeight):
    def __init__(self, ref_dist, sensor_dist, freq):
        super().__init__()
        
        self.ref_dist = ref_dist
        self.sensor_dist = sensor_dist
        self.freq = freq
    
    def heightmap(self, imgs):
        phase = self.phasemap(imgs)

        #heightmap = np.divide(self.ref_dist * phase_diff, 2.0 * np.pi * self.sensor_dist * self.freq)
        
        #heightmap[heightmap <= 0] = 0 # Remove negative values

        return None

class PolyPhaseHeight(PhaseHeight):
    def __init__(self, coeffs=None):
        super().__init__()
        
        self.coeffs = coeffs
    
    def calibrate(self, heightmap, ref_imgs, imgs, deg=1):
        ref_phase, measured_phase = self.phasemap(ref_imgs), self.phasemap(imgs)
        
        diff = ref_phase - measured_phase

        total = np.zeros(ref_phase.shape, dtype=np.float64)
        
        for i in range(deg):
            total += np.power(diff, i)

        # Coefficients are in ascending order

        self.coeffs, stats = P.polyfit(diff.ravel(), heightmap.ravel(), deg=deg, full=True)
            
        return self.coeffs, stats[0][0]

    def heightmap(self, ref_imgs, imgs):
        ref_phase, measured_phase = self.phasemaps(ref_imgs, imgs)
        
        diff = ref_phase - measured_phase
        
        result = np.zeros(ref_phase.shape, dtype=np.float64)
        
        for i, a_i in enumerate(self.coeffs):
            print(f'{round(a_i, ndigits=3)} X_{i}')
            result += (np.power(diff, i) * a_i)
        
        return result