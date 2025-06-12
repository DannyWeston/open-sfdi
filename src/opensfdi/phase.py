import cv2
import numpy as np

from abc import ABC, abstractmethod
from skimage.restoration import unwrap_phase

from . import image

# Phase Unwrapping

class PhaseUnwrap(ABC):
    def __init__(self, fringe_count):
        self.__fringe_count = fringe_count

    @abstractmethod
    def unwrap(self, phasemap, vertical=True):
        raise NotImplementedError

    def get_fringe_count(self) -> list[float]:
        return self.__fringe_count

class SpatialPhaseUnwrap(PhaseUnwrap):
    @abstractmethod
    def __init__(self, fringe_count):
        super().__init__([fringe_count])

class TemporalPhaseUnwrap(PhaseUnwrap):
    @abstractmethod
    def __init__(self, fringe_count):
        super().__init__(fringe_count)

class ReliabilityPhaseUnwrap(SpatialPhaseUnwrap):
    def __init__(self, fringe_count, wrap_around=False):
        super().__init__(fringe_count)

        self.wrap_around = wrap_around

    def unwrap(self, phasemap, vertical=True):
        return unwrap_phase(phasemap, wrap_around=self.wrap_around)

class MultiFreqPhaseUnwrap(PhaseUnwrap):
    def __init__(self, fringe_count):
        super().__init__(fringe_count)

    def unwrap(self, phasemaps):
        total = len(phasemaps)

        if total < 2:
            raise Exception("You must pass at least two spatial frquencies to use ")

        fringe_counts = self.get_fringe_count()

        # First phasemap is already unwrapped by definition
        # Lowest frequency phasemap has absolute frequency
        unwrapped = phasemaps[0].copy()

        for i in range(1, total):
            ratio = fringe_counts[i] / fringe_counts[i-1]

            k = np.round(((unwrapped * ratio) - phasemaps[i]) / (2.0 * np.pi))

            unwrapped = phasemaps[i] + (2.0 * np.pi * k)

        return unwrapped

# Phase Shifting

class PhaseShift(ABC):
    @abstractmethod
    def __init__(self, phase_count, shift_mask=0.01):
        self.__phase_count = phase_count

        self._shift_mask = shift_mask

    @abstractmethod
    def get_phases(self) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def shift(self, imgs) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def phase_count(self):
        return self.__phase_count
    
    @phase_count.setter
    def phase_count(self, value):
        self.__phase_count = value

    def get_phases(self):
        return np.linspace(0, 2.0 * np.pi, self.phase_count, endpoint=False)

class NStepPhaseShift(PhaseShift):
    def __init__(self, phase_count=3, shift_mask=0.0):
        super().__init__(phase_count, shift_mask=shift_mask)

        if phase_count < 3:
            raise Exception("The N-step method requires 3 or more phases")

    def shift(self, imgs) -> np.ndarray:
        a = np.zeros_like(imgs[0])
        b = np.zeros_like(a)

        phases = self.get_phases()
        N = len(imgs)
        
        # Check number of passed images is expected
        assert self.phase_count == N

        for i, phase in enumerate(phases):
            a += imgs[i] * np.sin(phase)
            b += imgs[i] * np.cos(phase)

        result = np.arctan2(a, b)
        result[result < 0] += 2.0 * np.pi # Correct arctan2 function

        if 0.0 < self._shift_mask:
            mod = (2.0 / N) * np.sqrt(a ** 2 + b ** 2)
            float_mask = image.threshold_mask(mod, threshold=self._shift_mask)
            float_mask[float_mask == 0.0] = np.nan # Set to nans

            result *= float_mask

        return result

def show_phasemap(phasemap, name='Phasemap'):
    # Mark nans as black

    norm = cv2.normalize(phasemap, None, 0.0, 1.0, cv2.NORM_MINMAX)
    image.show_image(norm, name)