from abc import ABC, abstractmethod

from ..image import ThresholdMask

from . import ShowPhasemap

from ..utils import ProcessingContext

class PhaseShift(ABC):
    @abstractmethod
    def __init__(self, phaseCounts):
        self.m_PhaseCounts = phaseCounts
    
    @property
    def phaseCounts(self):
        return self.m_PhaseCounts

    @abstractmethod
    def Shift(self, imgs):
        raise NotImplementedError

    def __iter__(self):
        return self.phaseCounts.__iter__()

    def __next__(self):
        return self.phaseCounts.__next__()

class NStepPhaseShift(PhaseShift):
    def __init__(self, phaseCounts, contrastMask=(0.0, 1.0)):
        super().__init__(phaseCounts=phaseCounts)

        self.m_ContrastMask = contrastMask

        if (v := len(phaseCounts)) < 3:
            raise Exception(f"The N-step method requires 3 or more phases ({v} passed)")

    def Shift(self, imgs):
        xp = ProcessingContext().xp

        N = len(imgs)

        # Generate phases (weird reshape for making sure it matches image channel count)
        phases = (xp.arange(N) * 2.0 * xp.pi) / N
        phases = phases.reshape(-1, *[1] * (imgs.ndim - 1))

        a = xp.sum(xp.sin(phases) * imgs, axis=0)
        b = xp.sum(xp.cos(phases) * imgs, axis=0)

        result = xp.arctan2(a, b)

        # ShowPhasemap(result, size=(1000, 1000))

        dcImage = xp.mean(imgs, axis=0)
 
        acImage = (2.0 / N) * xp.sqrt(a ** 2 + b ** 2)

        # Return result if no masking needed
        if self.m_ContrastMask is None:
            return result, dcImage

        mask = ThresholdMask(acImage, 
            min=self.m_ContrastMask[0], 
            max=self.m_ContrastMask[1]
        )
        
        mask[mask < 0.0] = xp.nan # Anything zero set to NaNs

        return result * mask, dcImage