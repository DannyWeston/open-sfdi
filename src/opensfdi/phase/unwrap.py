from abc import ABC, abstractmethod
from skimage.restoration import unwrap_phase

from . import ShowPhasemap

from ..utils import ProcessingContext

# Phase Unwrapping

class PhaseUnwrap(ABC):
    def __init__(self):
        self.m_Debug = False

    @property
    def debug(self):
        return self.m_Debug
    
    @debug.setter
    def debug(self, value):
        self.m_Debug = value

    @abstractmethod
    def Unwrap(self, phasemap, **kwargs):
        raise NotImplementedError

    @property
    def stripeCount(self):
        return self.m_StripeCount

class ReliabilityPhaseUnwrap(PhaseUnwrap):
    def __init__(self, stripeCount: float, wrapAround=False):
        super().__init__()

        self.m_WrapAround = wrapAround
        self.m_StripeCount = stripeCount

    def Unwrap(self, phasemap):
        return unwrap_phase(phasemap, wrap_around=self.m_WrapAround)
    
class MultiFreqPhaseUnwrap(PhaseUnwrap):
    def __init__(self, stripeCount):
        super().__init__()
        
        self.m_StripeCount = stripeCount

    def Unwrap(self, phasemaps):
        xp = ProcessingContext().xp
        
        total = len(phasemaps)
        if total < 2: raise Exception("You must pass at least two spatial frquencies to use")

        # Get correct number of stripes used per fringe direction
        numStripes = xp.asarray(self.stripeCount)
        phasemaps = xp.asarray(phasemaps)
 
        for i in range(1, total):
            ratio = numStripes[i] / numStripes[i-1]
            
            k = xp.round(((phasemaps[i-1] * ratio) - phasemaps[i]) / (2.0 * xp.pi))

            phasemaps[i] += 2.0 * xp.pi * k

        return phasemaps[-1]