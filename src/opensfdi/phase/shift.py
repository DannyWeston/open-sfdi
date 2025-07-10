import numpy as np

from abc import ABC, abstractmethod

from . import ShowPhasemap
from ..image import ThresholdMask, Show

class PhaseShift(ABC):
    @abstractmethod
    def __init__(self, phase_count, shift_mask=0.01):
        self.__phase_count = phase_count

        self._shift_mask = shift_mask

    @abstractmethod
    def GetPhases(self) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def Shift(self, imgs) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def phase_count(self):
        return self.__phase_count
    
    @phase_count.setter
    def phase_count(self, value):
        self.__phase_count = value

    def GetPhases(self):
        return np.linspace(0, 2.0 * np.pi, self.phase_count, endpoint=False)

class NStepPhaseShift(PhaseShift):
    def __init__(self, phase_count=3, mask=0.0):
        super().__init__(phase_count, shift_mask=mask)

        if phase_count < 3:
            raise Exception("The N-step method requires 3 or more phases")

    def Shift(self, imgs) -> np.ndarray:
        a = np.zeros_like(imgs[0])
        b = np.zeros_like(a)

        phases = self.GetPhases()
        N = len(imgs)
        
        # Check number of passed images is expected
        assert self.phase_count == N

        for i, phase in enumerate(phases):
            a += imgs[i] * np.sin(phase)
            b += imgs[i] * np.cos(phase)

        result = np.arctan2(-a, b)
        result[result < 0] += 2.0 * np.pi # Correct arctan2 function

        if self._shift_mask <= 0.0:
            return result
            
        mod = (2.0 / N) * np.sqrt(a ** 2 + b ** 2)

        float_mask = ThresholdMask(mod, threshold=self._shift_mask)
        float_mask[float_mask == 0.0] = np.nan # Set to nans

        return result * float_mask
