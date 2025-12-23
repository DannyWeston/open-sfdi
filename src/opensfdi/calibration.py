import numpy as np
import time
import cv2

from abc import ABC, abstractmethod

from .phase import ShowPhasemap, unwrap, shift
from .devices import camera, characterisation, projector
from .utils import ProcessingContext

from . import image, utils

class FPCharacteriser(ABC):
    @abstractmethod
    def __init__(self):
        # TODO: Allow for other calibration artefacts
        self.m_Debug = False

    @property
    def debug(self):
        return self.m_Debug
    
    @debug.setter
    def debug(self, value):
        self.m_Debug = value

class StereoCharacteriser(FPCharacteriser):
    def __init__(self, calibBoard: characterisation.CalibrationBoard, contrastMask=None):
        super().__init__()

        self.m_ContrastMask = contrastMask
        self.m_CalibBoard = calibBoard

        self.m_GatheringImages = False
        self.m_CaptureCallbacks = []

    def AddCaptureCallback(self, func):
        self.m_CaptureCallbacks.append(func)

    def RemoveCaptureCallback(self, func):
        self.m_CaptureCallbacks.remove(func)

    def FringeProject(self, cam: camera, proj: projector, stripeCount, phase, fringeRotation) -> image.Image:
        xp = ProcessingContext().xp

        # Display the fringes
        proj.numStripes = stripeCount
        proj.phase = phase
        proj.stripeRotation = fringeRotation
        proj.Display()

        # May want to add some noise for testing phase count number
        # rawData = image.AddGaussianNoise(rawData, sigma=0.03)
        # rawData = image.AddSaltPepperNoise(rawData, 0.00001, 0.00001)

        # Capture the camera data - use processing context for image
        return cam.Capture()

    def Measure(self, cam: camera, proj: projector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, fringeRotation=0.0, reverse=False):
        xp = ProcessingContext().xp

        shifted = xp.empty((len(unwrapper.stripeCount), *cam.shape), dtype=xp.float64)
        img = None

        for i, (numStripes, N) in enumerate(zip(unwrapper.stripeCount, shifter.phaseCounts)):

            imgs = xp.empty((N, *cam.shape), dtype=xp.float64)
            
            for j, phi in enumerate((xp.arange(N) * 2.0 * xp.pi) / N):

                tempImg = self.FringeProject(cam, proj, numStripes, phi, fringeRotation)

                imgData = tempImg.rawData

                if reverse:
                    imgs[(N-j) % N] = imgData
                else: 
                    imgs[j] = imgData

            shifted[i], dcImage, acImage = shifter.Shift(imgs)

            # Use lowest number of stripe shifted fringes for a DC image
            if i == 0: 
                img = dcImage

                # if self.m_ContrastMask: # Create a mask
                #     mask = image.ThresholdMask(dcImage,
                #         min=self.m_ContrastMask[0],
                #         max=self.m_ContrastMask[1]
                #     )

                #     mask[mask < 0.0] = xp.nan # Anything zero set to NaNs
                    
                #     dcImage *= mask
                #     shifted[i] *= mask

        # Calculate unwrapped phase maps
        phasemap = unwrapper.Unwrap(shifted)

        return phasemap, img
  
    def XYMeasure(self, cam, proj, shifter, unwrapper, yShifter=None, yUnwrapper=None, fringeRotation=0.0, dcFromX=True):
        xp = ProcessingContext().xp

        # Gather the phasemaps
        xPhasemap, dcImage = self.Measure(cam, proj, shifter, unwrapper, fringeRotation)
        # ShowPhasemap(xPhasemap, size=(1600, 900))

        if yShifter is None: yShifter = shifter
        if yUnwrapper is None: yUnwrapper = unwrapper

        yPhasemap, dcImage2 = self.Measure(cam, proj, yShifter, yUnwrapper, fringeRotation + np.pi) # ignore DC image
        # ShowPhasemap(yPhasemap, size=(1600, 900))

        return (dcImage if dcFromX else dcImage2), xPhasemap, yPhasemap

    # Private Functions

    def __RunAfterCaptureCallbacks(self, img, numImages):
        for func in self.m_CaptureCallbacks:
            func(img, numImages)

    def __ShowPhasemaps(self, xPhasemap, yPhasemap):
        winName = f"Characterisation"

        xPhasemap = utils.ToNumpy(xPhasemap)
        yPhasemap = utils.ToNumpy(yPhasemap)

        ShowPhasemap(xPhasemap, winName)
        ShowPhasemap(yPhasemap, winName)