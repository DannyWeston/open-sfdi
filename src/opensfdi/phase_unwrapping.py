from abc import ABC, abstractmethod

from skimage.restoration import unwrap_phase

class PhaseUnwrap(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def unwrap(self, phasemaps, *args, **kwargs):
        """ TODO: Write description """
        pass

class ReliabilityPhaseUnwrap(PhaseUnwrap):
    def __init__(self):
        """ TODO: Write description """
        pass

    def unwrap(self, phasemaps):
        """ TODO: Write description """
        # Simple passthrough to existing library
        return unwrap_phase(phasemaps)

class TemporalPhaseUnwrap(ABC):
    def __init__(self):
        pass