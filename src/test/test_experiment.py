import numpy as np
import cv2
from pathlib import Path

from opensfdi import wrapped_phase, unwrap_phase, show_surface
from opensfdi.profilometry import poly_phase_height
from opensfdi.experiment import NStepFPExperiment
from opensfdi.video import Camera, FringeProjector

class FakeCamera(Camera):
    def __init__(self):
        super().__init__()

        self.__resolution: tuple[int, int] = (1080, 1920)

    @property
    def resolution(self):
        """ (height, width) resolution of image captures (must be greater than zero) """
        return self.__resolution
    
    @resolution.setter
    def resolution(self, value):
        if any(k < 0 for k in value): raise ValueError("Resolution dimensions must be greater than zero")
        
        self.__resolution = value

    def capture(self):
        return np.ones(shape=(*self.resolution, 3), dtype=np.float32)

class FakeFP(FringeProjector):
    def __init__(self):
        super().__init__()

        self.__resolution : tuple[int, int] = (768, 1024)
        self.__frequency : float = 1
        self.__rotation : float = 1
        self.__phase : float = np.pi / 2.0

    @property
    def resolution(self):
        """ (height, width) resolution of image captures (must be greater than zero) """
        return self.__resolution
    
    @resolution.setter
    def resolution(self, value):
        if any(k < 0 for k in value): raise ValueError("Resolution dimensions must be greater than zero")
        
        self.__resolution = value

    @property
    def phase(self):
        return self.__phase
    
    @phase.setter
    def phase(self, value):
        self.__phase = value

    @property
    def frequency(self):
        return self.__frequency
    
    @frequency.setter
    def frequency(self, value):
        if value < 0: raise ValueError("Frequency must be greater than zero")
        self.__frequency = value

    @property
    def rotation(self):
        return self.__rotation
    
    @rotation.setter
    def rotation(self, value):
        self.__rotation = value

    def display(self):
        pass

def test_nstep():
    # Load the images from disk
    input_dir = Path('C:\\Users\\danie\\Desktop\\Results\\Test1')

    # Load the calibration images
    calib_dir = input_dir / "Calibration"

    calib_count = 41
    calib_phases = 3
    calib_images = []
    calib_dists = np.linspace(0.0, 200.0, 41, dtype=np.float64)

    for i in range(calib_count):
        calib_images.append(np.array([cv2.imread(calib_dir / f"{i+1}_{j+1}.jpg") for j in range(calib_phases)]))

    calib_images = np.array(calib_images)[::2]
    calib_dists = calib_dists[::2]

    poly_deg = 5
    coeffs = poly_phase_height(calib_images, calib_dists, poly_deg)
    print(coeffs.shape)

    # Now have the coefficients for height, so can use for construction

    # Load the measurement images
    meas_dir = input_dir / "Measurement"
    meas_phases = 12

    ref_imgs = []
    meas_imgs = []

    for i in range(meas_phases):
        ref_imgs.append(cv2.imread(meas_dir / f"ref_{i}.jpg"))
        meas_imgs.append(cv2.imread(meas_dir / f"measurement_{i}.jpg"))

    # Calculate phasemap
    ref_phase = unwrap_phase(wrapped_phase(ref_imgs))
    measured_phase = unwrap_phase(wrapped_phase(meas_imgs))
    phase_diff = measured_phase - ref_phase

    height = coeffs[0]
    for i in range(1, poly_deg):
        height += np.power(phase_diff, i) * coeffs[i]

    show_surface(height)

    assert False

def test_fringe_projection():
    p = FakeFP()
    c = FakeCamera()

    expected_img = c.capture()
    
    for i in range(3, 10):
        print(f"Testing {i}-step methodology")

        ref_imgs, imgs = NStepFPExperiment([c], p, i).run()

        for cam in ref_imgs: # Test actual images
            for img in cam:
                assert (img == expected_img).all()

        for cam in imgs: # Test measurement images
            for img in cam:
                assert (img == expected_img).all()

    assert True

# image = FringeFactory.MakeSinusoidal(1, 3, -np.pi / 2, width=4, height=4)[0]
# #image = color.rgb2gray(img_as_float(data.chelsea()))

# vals = np.array([
#     np.array([2.1, 0.5, 0.2, 0.1]),
#     np.array([0.3, 1.0, 1.9, 0.3]),
#     np.array([1.6, 0.9, 0.2, 0.3]),
#     np.array([0.1, 0.4, 0.1, 2.0]),
# ])

# image = exposure.rescale_intensity(image, out_range=(0, 2 * np.pi))
# #show_phasemap(image)

# # Create a phase-wrapped image in the interval [-pi, pi)
# image_wrapped = np.angle(np.exp(1j * image))
# #show_phasemap(image_wrapped)

# # Perform phase unwrapping
# image_unwrapped = itoh.unwrap_phase(image_wrapped)
# #show_phasemap(image_unwrapped)

# image_unwrapped_2 = reliability.unwrap_phase(image_wrapped)
# show_phasemap(image_unwrapped_2)

# # Perform phase unwrapping
# #image_unwrapped_3 = rest.unwrap_phase(image_wrapped)
# #show_phasemap(image_unwrapped_3)

# fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
# ax1, ax2, ax3, ax4 = ax.ravel()

# fig.colorbar(ax1.imshow(image, cmap='gray', vmin=0, vmax=4 * np.pi), ax=ax1)
# ax1.set_title('Original')

# fig.colorbar(ax2.imshow(image_wrapped, cmap='gray', vmin=-np.pi, vmax=np.pi), ax=ax2)
# ax2.set_title('Wrapped phase')

# fig.colorbar(ax3.imshow(image_unwrapped, cmap='gray'), ax=ax3)
# ax3.set_title('Simple phase unwrapping')

# fig.colorbar(ax4.imshow(image_unwrapped_2, cmap='gray'), ax=ax4)
# ax4.set_title('Scikit phase unwrapping')

# plt.show()