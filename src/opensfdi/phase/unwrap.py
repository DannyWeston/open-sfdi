import numpy as np

from abc import ABC, abstractmethod
from skimage.restoration import unwrap_phase

from . import ShowPhasemap

from ..utils import ProcessingContext

# Phase Unwrapping

class PhaseUnwrap(ABC):
    def __init__(self, numStripesVertical, numStripesHorizontal=None):
        self.m_NumStripesVertical = numStripesVertical

        # Allow for different horizontal stripes count to be passed
        self.m_NumStripesHorizontal = numStripesVertical if numStripesHorizontal is None else numStripesHorizontal

        self.m_Vertical = True

    @property
    def vertNumStripes(self):
        return self.m_NumStripesVertical
    
    @property
    def horiNumStripes(self):
        return self.m_NumStripesHorizontal

    @property
    def vertical(self):
        return self.m_Vertical
    
    @vertical.setter
    def vertical(self, value: bool):
        self.m_Vertical = value

    @abstractmethod
    def Unwrap(self, phasemap):
        raise NotImplementedError

    @property
    def stripeCount(self):
        return self.vertNumStripes if self.vertical else self.horiNumStripes

    def __iter__(self):
        return self.stripeCount.__iter__()

    def __next__(self):
        return self.stripeCount.__next__()

class SpatialPhaseUnwrap(PhaseUnwrap):
    @abstractmethod
    def __init__(self, numStripesVertical, numStripesHorizontal=None):
        super().__init__(numStripesVertical, numStripesHorizontal=numStripesHorizontal)

class TemporalPhaseUnwrap(PhaseUnwrap):
    @abstractmethod
    def __init__(self, numStripesVertical, numStripesHorizontal=None):
        super().__init__(numStripesVertical, numStripesHorizontal=numStripesHorizontal)

class ReliabilityPhaseUnwrap(SpatialPhaseUnwrap):
    def __init__(self, numStripesVertical, numStripesHorizontal=None, wrapAround=False):
        super().__init__(numStripesVertical, numStripesHorizonta=numStripesHorizontal)

        self.m_WrapAround = wrapAround

    def Unwrap(self, phasemap):
        return unwrap_phase(phasemap, wrap_around=self.m_WrapAround)

class MultiFreqPhaseUnwrap(PhaseUnwrap):
    def __init__(self, numStripesVertical, numStripesHorizontal=None):
        super().__init__(numStripesVertical, numStripesHorizontal=numStripesHorizontal)

    def Unwrap(self, phasemaps):
        xp = ProcessingContext().xp
        
        total = len(phasemaps)

        if total < 2: raise Exception("You must pass at least two spatial frquencies to use ")
 
        numStripes = xp.asarray(self.stripeCount)
        phasemaps = xp.asarray(phasemaps)

        unwrapped = phasemaps[0]

        ratios = numStripes[1:] / numStripes[:-1]

        for i in range(1, total):
            k = xp.round(((unwrapped * ratios[i-1]) - phasemaps[i]) / (2.0 * xp.pi))

            unwrapped = phasemaps[i] + (2.0 * xp.pi * k)

        return unwrapped