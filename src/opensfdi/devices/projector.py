import numpy as np
import cv2

from abc import abstractmethod

from .characterisation import Characterisation, ICharacterisable, CalibrationBoard
from .. import utils

class Projector(ICharacterisable):
    @abstractmethod
    def __init__(self, resolution, channels, refreshRate, throwRatio, aspectRatio, character:Characterisation=None):
        self.m_Resolution = resolution
        self.m_Channels = channels
        self.m_RefreshRate = refreshRate

        self.m_AspectRatio = aspectRatio
        self.m_ThrowRatio = throwRatio

        self.m_Characterisation = character

        self.m_ShouldUndistort = True
        self.m_Debug = False

    @property
    def characterisation(self) -> Characterisation:
        return self.m_Characterisation

    @property
    def resolution(self) -> tuple[int, int]:
        return self.m_Resolution
    
    @property
    def channels(self) -> int:
        return self.m_Channels
    
    @property
    def refreshRate(self) -> float:
        return self.m_RefreshRate

    @property
    def throwRatio(self) -> float:
        return self.m_ThrowRatio
    
    @property
    def aspectRatio(self) -> float:
        return self.m_AspectRatio
    
    @property
    def shape(self):
        if self.channels == 1:
            return self.resolution
        
        return (*self.resolution, self.channels)

    @property
    def debug(self):
        return self.m_Debug
    
    @debug.setter
    def debug(self, value):
        self.m_Debug = value

    @abstractmethod
    def Display(self):
        raise NotImplementedError

class FringeProjector(Projector):
    def __init__(self, resolution, channels, refreshRate, throwRatio, aspectRatio, 
                stripeRotation, phase, numStripes,
                character: Characterisation=None):
        
        super().__init__(resolution, channels, refreshRate, throwRatio, aspectRatio, character=character)

        self.m_StripeRotation = stripeRotation
        self.m_Phase = phase
        self.m_NumStripes = numStripes

    @property
    def numStripes(self) -> float:
        return self.m_NumStripes
    
    @numStripes.setter
    def numStripes(self, value):
        if value < 0: return
        
        self.m_NumStripes = value

    @property
    def phase(self) -> float:
        return self.m_Phase

    @phase.setter
    def phase(self, value):
        self.m_Phase = value

    @property
    def stripeRotation(self) -> float:
        return self.m_StripeRotation

    @stripeRotation.setter
    def stripeRotation(self, value):
        self.m_StripeRotation = value

    def PhaseToCoord(self, cameraCoords, xPhasemap, yPhasemap, xNumStripes, yNumStripes, bilinear=True) -> np.ndarray:
        xp = utils.ProcessingContext().xp

        h, w = self.resolution

        projCoords = xp.empty_like(cameraCoords)

        periodX = w / xNumStripes
        periodY = h / yNumStripes

        for i, coords in enumerate(cameraCoords):
            # opencv shitty x-y convention for characterisation needs to be switched back to y-x for images
            flipped = xp.flip(coords)

            if bilinear:
                phiX = utils.SingleBilinearInterp(xPhasemap, flipped) # x-coord from xPhasemap
                phiY = utils.SingleBilinearInterp(yPhasemap, flipped) # y-coord from yPhasemap

            else:
                flipped = flipped.astype(xp.uint16)
                phiX = xPhasemap[*flipped]
                phiY = yPhasemap[*flipped]
                
            projCoords[i, 0] = phiX / xp.pi / 2.0 * periodX
            projCoords[i, 1] = phiY / xp.pi / 2.0 * periodY

        return projCoords

    def Characterise(self, board: CalibrationBoard, poiCoords, extraFlags=0):
        # extraFlags += cv2.CALIB_FIX_ASPECT_RATIO
        # extraFlags += cv2.CALIB_FIX_PRINCIPAL_POINT

        return self.characterisation.Calculate(board, poiCoords, 
            self.resolution, extraFlags=extraFlags
        )

    @abstractmethod
    def Display(self):
        raise NotImplementedError