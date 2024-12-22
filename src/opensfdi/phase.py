import numpy as np
import pydantic

from typing import Optional, Callable

from abc import ABC, abstractmethod
from skimage.restoration import unwrap_phase

# Phase Unwrapping

class PhaseUnwrap(pydantic.BaseModel, ABC):
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not issubclass(v, PhaseShift):
            raise ValueError("Invalid Object")

        return v

    @abstractmethod
    def unwrap(self, phasemaps, *args, **kwargs):
        """ TODO: Write description """
        pass

class ReliabilityPhaseUnwrap(PhaseUnwrap):
    wrap_around: bool = False

    def unwrap(self, phasemaps):
        """ TODO: Write description """
        # Simple passthrough to existing library
        return unwrap_phase(phasemaps, wrap_around=self.wrap_around)

class TemporalPhaseUnwrap(PhaseUnwrap):
    pass


# Phase Shifting

class PhaseShift(pydantic.BaseModel, ABC):
    phase_count: int = 3

    model_config = pydantic.ConfigDict(extra='allow', arbitrary_types_allowed=True)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not issubclass(v, PhaseShift):
            raise ValueError("Invalid Object")

        return v

    @abstractmethod
    def get_phases(self) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def shift(self, imgs, *args, **kwargs):
        if imgs is None: raise TypeError

        if len(imgs) == 0: raise ValueError("Images passed has length zero")

class NStepPhaseShift(PhaseShift):
    def shift(self, imgs):
        super().shift(imgs)

        p = np.zeros_like(imgs[0], dtype=np.float64)
        q = np.zeros_like(imgs[0], dtype=np.float64)

        # Accumulate phases
        for phase, img in zip(self.get_phases(), imgs, strict=True):
            p += img * np.sin(phase)
            q += img * np.cos(phase)

        return -np.arctan2(p, q)
    
    def get_phases(self) -> list[np.ndarray]:
        return np.arange(self.phase_count) * 2.0 * np.pi / self.phase_count