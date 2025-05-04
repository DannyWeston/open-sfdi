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

    def unwrap(self, phasemaps, vertical=True):
        total = len(phasemaps)

        if total < 2:
            raise Exception("You must pass at least two spatial frquencies to use ")

        # First phasemap is already unwrapped by definition
        unwrapped = np.empty_like(phasemaps)

        fringe_counts = self.get_fringe_count()

        h, w = phasemaps[0].shape
        fringe_freqs = np.array(fringe_counts) / (w if vertical else h)

        # Lowest frequency phasemap has absolute frequency
        unwrapped[0] = unwrap_phase(phasemaps[0])

        for i in range(1, total):
            phi = phasemaps[i]

            estimate = (fringe_freqs[i] / fringe_freqs[i-1]) * unwrapped[i-1]

            # Calculate fringe order
            fringe_order = np.round((estimate - phi) / (2.0 * np.pi))

            # Accumulate phases
            unwrapped[i] = phi + (fringe_order * 2.0 * np.pi)

        return unwrapped[-1]


# Phase Shifting

class PhaseShift(ABC):
    @abstractmethod
    def __init__(self, phase_count):
        self.__phase_count = phase_count

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

class NStepPhaseShift(PhaseShift):
    def __init__(self, phase_count=3):
        super().__init__(phase_count)

        if phase_count < 3:
            raise Exception("The N-step method requires 3 or more phases")

    def shift(self, imgs) -> np.ndarray:
        a = np.zeros(shape=imgs[0].shape, dtype=np.float32)
        b = np.zeros_like(a)

        phases = self.get_phases()

        # Check number of passed images is expected
        assert self.phase_count == len(imgs)

        for i, phase in enumerate(phases):
            a += imgs[i] * np.sin(phase)
            b += imgs[i] * np.cos(phase)

        return np.arctan2(a, b)

    def get_phases(self):
        return np.linspace(0, 2.0 * np.pi, self.phase_count, endpoint=False)
    
def show_phasemap(phasemap, name='Phasemap'):
    norm_data = cv2.normalize(phasemap, None, 0.0, 1.0, cv2.NORM_MINMAX)

    image.show_image(norm_data, name)