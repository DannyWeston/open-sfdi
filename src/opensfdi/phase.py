import numpy as np

from abc import ABC, abstractmethod
from skimage.restoration import unwrap_phase

# Phase Unwrapping

class PhaseUnwrap(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def unwrap(self, phasemaps, *args, **kwargs):
        """ TODO: Write description """
        pass

class ReliabilityPhaseUnwrap(PhaseUnwrap):
    def __init__(self, wrap_around=False):
        """ TODO: Write description """
        self.__wrap_around = wrap_around

    def unwrap(self, phasemaps):
        """ TODO: Write description """
        # Simple passthrough to existing library
        return unwrap_phase(phasemaps, wrap_around=self.__wrap_around)

class TemporalPhaseUnwrap(ABC):
    def __init__(self):
        pass


# Phase Shifting

class PhaseShift(ABC):
    @abstractmethod
    def __init__(self, required_imgs: int):
        if required_imgs <= 0:
            raise ValueError("Required images less than or equal to zero")
        
        # raise Exception(f"You need at least 3 phases to run an N-step experiment ({steps} provided)")

        self.__required_imgs = required_imgs

    @property
    def required_imgs(self):
        return self.__required_imgs
    
    @abstractmethod
    def get_phases(self):
        pass

    @abstractmethod
    def shift(self, imgs, *args, **kwargs):
        if imgs is None: raise TypeError

        if len(imgs) == 0: raise ValueError("Images passed has length zero")

class NStepPhaseShift(PhaseShift):
    def __init__(self, steps):
        super().__init__(steps)

    def shift(self, imgs, steps=3):
        super().shift(imgs)

        if steps < 3: raise Exception(f"You need at least 3 phases to run an N-step experiment ({steps} provided)")

        N = len(imgs)

        p = np.zeros(shape=imgs[0].shape, dtype=np.float64)
        q = np.zeros(shape=imgs[0].shape, dtype=np.float64)
        
        # Accumulate phases
        for i, img in enumerate(imgs):
            phase = (2.0 * np.pi * i) / N

            p += img * np.sin(phase)
            q += img * np.cos(phase)

        return -np.arctan2(p, q)
    
    def get_phases(self):
        return np.arange(self.required_imgs) * 2.0 * np.pi / self.required_imgs