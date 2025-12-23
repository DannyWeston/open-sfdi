from abc import ABC, abstractmethod

from ..image import Show, ThresholdMask

from . import ShowPhasemap

from .. import utils

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
    def __init__(self, phaseCounts):
        super().__init__(phaseCounts=phaseCounts)


        if (v := len(phaseCounts)) < 3:
            raise Exception(f"The N-step method requires 3 or more phases ({v} passed)")

    def Shift(self, imgs):
        xp = utils.ProcessingContext().xp

        N = len(imgs)

        # Generate phases (weird reshape for making sure it matches image channel count)
        phases = xp.arange(N) * 2.0 * xp.pi / N

        # phases = phases.reshape(-1, *[1] * (imgs.ndim - 1))

        a = xp.zeros_like(imgs[0])
        b = xp.zeros_like(imgs[0])

        for i, img in enumerate(imgs):
            p = phases[(i + (N // 2)) % N]

            a -= img * xp.sin(p)
            b += img * xp.cos(p)
            # ShowPhasemap(xp.arctan2(a, b), size=(1600, 900))

        result = xp.arctan2(a, b)
        
        # I have no clue why this is needed, but.... it fixes a problem?
        # I think its because the phasemaps need to be greater than 0 for correct PhaseToCoord conversion
        # aka convert form -pi <= result < pi to 0 <= result < 2pi
        result -= xp.min(result)

        dcImage = xp.mean(imgs, axis=0)

        acImage = (2.0 / N) * xp.sqrt(a ** 2 + b ** 2)

        # ShowPhasemap(result, size=(1600, 900))

        return result, dcImage, acImage