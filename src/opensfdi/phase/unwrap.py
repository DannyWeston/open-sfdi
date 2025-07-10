import numpy as np

from . import ShowPhasemap

from abc import ABC, abstractmethod
from skimage.restoration import unwrap_phase

# Phase Unwrapping

class PhaseUnwrap(ABC):
    def __init__(self, fringe_count):
        self.__fringe_count = fringe_count

    @abstractmethod
    def Unwrap(self, phasemap, vertical=True):
        raise NotImplementedError

    def GetFringeCount(self) -> list[float]:
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

    def Unwrap(self, phasemap, vertical=True):
        return unwrap_phase(phasemap, wrap_around=self.wrap_around)

class MultiFreqPhaseUnwrap(PhaseUnwrap):
    def __init__(self, fringe_count):
        super().__init__(fringe_count)

    def Unwrap(self, phasemaps):
        total = len(phasemaps)

        if total < 2:
            raise Exception("You must pass at least two spatial frquencies to use ")

        fringe_counts = self.GetFringeCount()

        # First phasemap is already unwrapped by definition
        # Lowest frequency phasemap has absolute frequency
        unwrapped = phasemaps[0].copy()

        for i in range(1, total):

            ratio = fringe_counts[i] / fringe_counts[i-1]

            k = np.round(((unwrapped * ratio) - phasemaps[i]) / (2.0 * np.pi))

            unwrapped = phasemaps[i] + (2.0 * np.pi * k)

        return unwrapped