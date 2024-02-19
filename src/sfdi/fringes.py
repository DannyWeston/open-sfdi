import cv2
import numpy as np

from sfdi.definitions import FRINGES_DIR

class Fringes:
    def binary(width, height, freq, orientation, phase, rgba=True):
        """
        Creates a binary fringe pattern, similar to a PWM signal.

        Args:
            width (int): Width of the fringe pattern (pixels).
            height (int): Height of the fringe pattern (pixels).
            freq (float): Spatial frequency of the fringes.
            orientation (float): Orientation of the fringes (0 = horizontal, 2pi = vertical).

        Returns:
            ndarray[uint8]: Binary fringe pattern.
        """ 

        # Maybe not a good idea to rely upon sinusoidal function
        # But works for now :)

        img = Fringes.sinusoidal(width, height, freq, phase, orientation)
        width, height = img.shape
        if rgba:
            for col in range(width):
                for row in range(height):
                    img[col][row][:] = 0 if img[col][row][0] < 0 else 1
        else:
            for col in range(width):
                for row in range(height):
                    img[col][row] = 0 if img[col][row] < 0 else 1

        return img #img.astype(np.uint8)

    def sinusoidal(width, height, freq, orientation, phase, rgba=True):
        """
        Creates a sinusoidal fringe pattern (values between -1 and 1)

        Args:
            width (int): Width of the fringe pattern (pixels).
            height (int): Height of the fringe pattern (pixels).
            freq (float): Spatial frequency of the fringes.
            orientation (float): Orientation of the fringes (0 = horizontal, 2pi = vertical).
    
        Returns:
            ndarray[float32]: Sinusoidal fringe pattern.
        """
        x, y = np.meshgrid(np.arange(width, dtype=int), np.arange(height, dtype=int))

        gradient = np.sin(orientation) * x - np.cos(orientation) * y
        
        img = np.sin(((2.0 * np.pi * gradient) / freq) + phase)
        
        img = (img - img.min()) / (img.max() - img.min())   # Normalise
        img = (img * 255.0).astype(np.uint8)                # Convert to uint8 dtype
        
        if rgba: return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def __init__(self, fringe_imgs):
        self._images = fringe_imgs

    def __iter__(self):
        return iter(self._images)

    def __next__(self):
        return next(self._images)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        return self._images[item]

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        self._images = images

    def from_file(names, img_repo):
        return Fringes([img_repo.load(name) for name in names])

    def from_generator(width, height, freq, orientation=(np.pi / 2.0), n=3, fringe_type='Sinusoidal'):
        """
        Generates a collection of fringe images with phase = 2pi * k / n for a given n,
        such that k ∈ {1..n}.
    
        Args:
            f_type (function): Type of fringe pattern to generate.
            width (int): Width of image.
            height (int): Height of image.
            freq (float): Spatial frequency of the fringes.
            orientation (float): Orientation of the fringes (0 = horizontal, 0.5pi = vertical).
            n (int): Number of different phases.
    
        Returns:
            list[nd.array]: List of n fringe patterns (images). 
        """
        
        if fringe_type == 'Sinusoidal': gen_func = Fringes.sinusoidal
        elif fringe_type == 'Binary': gen_func = Fringes.binary
        else: raise Exception("Incorrect fringe generator function provided")

        imgs = [gen_func(width, height, freq, orientation, (2.0 * i * np.pi) / n) for i in range(n)]

        return Fringes(imgs)

    def to_rgb(self):
        if self._images is None or len(self._images) == 0:
            return

        if self._images[0].shape[2] == 3: # Already rgb
            return

        self._images = [cv2.merge(img) for img in self._images]

# Useful sf values for certain projectors with properties: 
#   - Resolution:                           1024x1024
#   - Number of line pairs:                 8
#   - Size:                                 1m x 1m
#   - Projector tilt:                       10 degrees
#   - Projector distance to baseline stage: 0.5m
#   - Projector FOV:                        -
#   - Projection Width:                     0.64m
projector_sfs = {
    'LG' : 32,
    'BLENDER' : 16
}