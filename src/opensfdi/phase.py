import numpy as np

from abc import ABC, abstractmethod
from skimage.restoration import unwrap_phase

from . import image, utils

# Phase Shifting

class Shifter(ABC):
    @abstractmethod
    def __init__(self, phaseCounts):
        self.m_PhaseCounts = phaseCounts
    
    @property
    def phase_counts(self):
        return self.m_PhaseCounts

    @abstractmethod
    def shift(self, imgs):
        raise NotImplementedError

    def __iter__(self):
        return self.phase_counts.__iter__()

    def __next__(self):
        return self.phase_counts.__next__()

class NStepPhaseShift(Shifter):
    def __init__(self, phaseCounts):
        super().__init__(phaseCounts=phaseCounts)

        if (v := len(phaseCounts)) < 3:
            raise Exception(f"The N-step method requires 3 or more phases ({v} passed)")

    def shift(self, imgs):
        xp = utils.ProcessingContext().xp

        N = len(imgs)

        # Generate phases (weird reshape for making sure it matches image channel count)
        phases = xp.arange(N) * 2.0 * xp.pi / N

        sin_phases = xp.sin(phases)
        cos_phases = xp.cos(phases)

        a = xp.zeros_like(imgs[0])
        b = xp.zeros_like(imgs[0])

        for i, img in enumerate(imgs):
            a += img * sin_phases[i]
            b -= img * cos_phases[i]

        ac_img = (2.0 / N) * xp.sqrt(a ** 2 + b ** 2)
        dc_img = xp.mean(imgs, axis=0)

        result = -xp.arctan2(a, b)
        result += xp.pi

        return result, ac_img, dc_img

# Phase Unwrapping

class Unwrapper(ABC):
    def __init__(self, stripe_count):
        self._debug = False
        self._stripe_count = stripe_count

    @property
    def stripe_count(self):
        return self._stripe_count

    @property
    def debug(self):
        return self._debug
    
    @debug.setter
    def debug(self, value):
        self._debug = value

    @abstractmethod
    def Unwrap(self, phasemap, **kwargs):
        raise NotImplementedError

class ReliabilityPhaseUnwrap(Unwrapper):
    def __init__(self, stripe_count, wrapAround=False):
        super().__init__(stripe_count)

        self.m_WrapAround = wrapAround

    def Unwrap(self, phasemap):
        return unwrap_phase(phasemap, wrap_around=self.m_WrapAround)

class MultiFreqPhaseUnwrap(Unwrapper):
    def __init__(self, stripe_count):
        super().__init__(stripe_count)

    def Unwrap(self, phasemaps):
        xp = utils.ProcessingContext().xp
        
        total = len(phasemaps)
        if total < 2: raise Exception("You must pass at least two spatial frquencies to use")

        # Get correct number of stripes used per fringe direction
        
        for i in range(1, total):
            ratio = self._stripe_count[i] / self._stripe_count[i-1]

            # Original implementation, less memory efficient
            # scaled = phasemaps[i-1] * ratio
            # k = xp.round((scaled - phasemaps[i]) / (2.0 * xp.pi))
            # phasemaps[i] += k * 2.0 * xp.pi

            # Memory efficient way of completing the above
            phasemaps[i-1] *= ratio
            phasemaps[i-1] -= phasemaps[i]
            phasemaps[i-1] /= 2.0 * xp.pi
            xp.round(phasemaps[i-1], out=phasemaps[i-1])
            phasemaps[i] += phasemaps[i-1] * 2.0 * xp.pi

        return phasemaps[total-1]


# Utils

def show_phasemap(phasemap, name='Phasemap', size=None):
    with utils.ProcessingContext.UseGPU(False):
        image.show_img(image.Normalise(phasemap), name, size=size)