from abc import ABC, abstractmethod

from ..image import ThresholdMask, Show

from ..utils import ProcessingContext

class PhaseShift(ABC):
    @abstractmethod
    def __init__(self, phiVertical, phiHorizontal=None, contrastMask=0.01):
        self.m_PhiVertical = phiVertical
        self.m_PhiHorizontal = phiVertical if phiHorizontal is None else phiHorizontal

        self.m_ContrastMask = contrastMask

        self.m_Vertical = True

    @property
    def vertical(self):
        return self.m_Vertical
    
    @vertical.setter
    def vertical(self, value: bool):
        self.m_Vertical = value
    
    @property
    def phaseCounts(self):
        return self.m_PhiVertical if self.vertical else self.m_PhiHorizontal

    @abstractmethod
    def Shift(self, imgs):
        raise NotImplementedError

    def __iter__(self):
        return self.phaseCounts.__iter__()

    def __next__(self):
        return self.phaseCounts.__next__()

class NStepPhaseShift(PhaseShift):
    def __init__(self, phiVertical, phiHorizontal=None, mask=0.0):
        super().__init__(phiVertical, phiHorizontal=phiHorizontal, contrastMask=mask)

        if (v := len(phiVertical)) < 3:
            raise Exception(f"The N-step method requires 3 or more phases ({v} passed for vertical)")
        
        if (phiHorizontal is not None) and ((h := len(phiHorizontal)) < 3):
            raise Exception(f"The N-step method requires 3 or more phases ({h} passed for horizontal)")

    def Shift(self, imgs):
        xp = ProcessingContext().xp
        
        N = len(imgs)

        # Generate phases (weird reshape for making sure it matches image channel count)
        phases = (xp.arange(N) * 2.0 * xp.pi) / N
        phases = phases.reshape(-1, *[1] * (imgs.ndim - 1))

        a = xp.sum(xp.sin(phases) * imgs, axis=0)
        b = xp.sum(xp.cos(phases) * imgs, axis=0)

        result = xp.arctan2(a, b)
        result[result < 0.0] += (xp.pi * 2.0)

        dcImage = (2.0 / N) * xp.sqrt(a ** 2 + b ** 2)

        # Return result if no masking needed
        if self.m_ContrastMask is None:
            return result, dcImage

        contrastMask = ThresholdMask(dcImage, threshold=self.m_ContrastMask)
        contrastMask[contrastMask == 0.0] = xp.nan # Anything zero set to NaNs

        return result * contrastMask, dcImage