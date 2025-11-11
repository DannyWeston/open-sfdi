import numpy as np
import time
import cv2 # TODO: Better

from abc import ABC, abstractmethod

from .phase import ShowPhasemap, unwrap, shift
from .devices import camera, projector, board, vision
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
    def __init__(self, calibBoard: board.CalibrationBoard):
        super().__init__()

        self.m_CalibBoard = calibBoard

        self.m_UseChannel = 0

        self.m_GatheringImages = False
        self.m_CaptureCallbacks = []

    @property
    def useChannel(self):
        return self.m_UseChannel
    
    @useChannel.setter
    def useChannel(self, value):
        self.m_UseChannel = value

    def AddCaptureCallback(self, func):
        self.m_CaptureCallbacks.append(func)

    def RemoveCaptureCallback(self, func):
        self.m_CaptureCallbacks.remove(func)

    def GatherPhasemap(self, camera: camera.Camera, projector: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap):
        xp = ProcessingContext().xp
        
        # TODO: Add some flag to save the images whilst gathering?
        stripeCount = unwrapper.stripeCount

        shifted = xp.empty((len(stripeCount), *camera.shape), dtype=xp.float32)

        dcImage = None

        for i, (numStripes, N) in enumerate(zip(stripeCount, shifter.phaseCounts)):

            imgs = xp.empty(shape=(N, *camera.shape), dtype=xp.float32)

            phases = (xp.arange(N) * 2.0 * xp.pi) / N

            # TODO: Vectorise
            for j in range(N):
                projector.numStripes = numStripes
                projector.phase = phases[j]
                projector.Display()

                rawData = camera.Capture().rawData

                # Use processing context for image
                rawData = xp.asarray(rawData)

                # Add some noise for testing phase count number
                # rawData = image.AddGaussianNoise(rawData, sigma=0.03)
                # rawData = image.AddSaltPepperNoise(rawData, 0.00001, 0.00001)

                imgs[j] = rawData

            # Use lowest frequency shifted fringes for DC image
            if i == 0: shifted[i], dcImage = shifter.Shift(imgs)
            else: shifted[i], _ = shifter.Shift(imgs)

        # Calculate unwrapped phase maps
        phasemap = unwrapper.Unwrap(shifted)

        return phasemap, dcImage

    def Characterise(self, cam: camera.Camera, proj: projector.FringeProjector, 
        xShifter: shift.PhaseShift, yShifter: shift.PhaseShift,
        xUnwrapper: unwrap.PhaseUnwrap, yUnwrapper: unwrap.PhaseUnwrap):

        if self.m_GatheringImages:
            raise Exception("A characterisation is already in progress")

        xp = ProcessingContext().xp

        dcImages = []

        cameraPOIs = []
        xPhasemaps = []
        yPhasemaps = []

        self.m_GatheringImages = True

        # For debugging
        projImg = np.zeros(shape=(*proj.resolution, 3), dtype=np.float32)

        while self.m_GatheringImages:
            tempTime = time.time()

            # Gather the phasemaps
            proj.stripeRotation = 0.0
            xPhasemap, dcImage = self.GatherPhasemap(cam, proj, xShifter, xUnwrapper)

            # ShowPhasemap(xPhasemap, size=(1000, 1000))

            proj.stripeRotation = np.pi / 2.0
            yPhasemap, _ = self.GatherPhasemap(cam, proj, yShifter, yUnwrapper) # ignore DC image

            poiCoords = self.m_CalibBoard.FindPOIS(dcImage)

            if poiCoords is None: # Didn't work, skip image
                print(f"Could not identify POIs, skipping image")
                continue
        
            # Found coords, store results for later
            print(f"Successfully identified POIs ({time.time() - tempTime:.3f}s)")

            dcImages.append(dcImage)
            cameraPOIs.append(poiCoords)

            # POIs for camera and projector successfully detected
            # TODO: Add for colour selection or averaging (and not using only useChannel)
            if xPhasemap.ndim == 3:
                xPhasemaps.append(xPhasemap[:, :, self.useChannel])
                yPhasemaps.append(yPhasemap[:, :, self.useChannel])
            else:
                xPhasemaps.append(xPhasemap)
                yPhasemaps.append(yPhasemap)

            # Run any post-capture callbacks
            self.__RunAfterCaptureCallbacks(dcImage, len(dcImages))

        print(f"{len(cameraPOIs)} total valid images captured")

        # We can now characterise the camera
        cameraPOIs = xp.asarray(cameraPOIs)
        boardCoords = self.m_CalibBoard.GetPOICoords()

        objectCoords = xp.repeat(boardCoords[xp.newaxis, ...], len(cameraPOIs), axis=0)

        camReprojErrors, camRMSError = cam.Characterise(objectCoords, cameraPOIs)
        print(f"Camera reprojection error: {camRMSError}")

        if self.debug:
            camNormErrors = (camReprojErrors - np.min(camReprojErrors)) / np.ptp(camReprojErrors)
            for i in range(len(cameraPOIs)):
                self.__ShowPOIs(dcImages[i].copy(), cameraPOIs[i], colourBy=camNormErrors[i])

        # Try to convert the camera POIs coords to projector coords using phase matching
        # In human terms: the camerea helps the projector "see" the characterisation board
        projPOIs = xp.empty_like(cameraPOIs)
        for i in range(len(projPOIs)):
            # Undistort the images with the now-calibrated camera
            dcImage = cam.Undistort(dcImages[i])
            xPhasemap = cam.Undistort(xPhasemaps[i])
            yPhasemap = cam.Undistort(yPhasemaps[i])

            poiCoords = self.m_CalibBoard.FindPOIS(dcImage)

            # Complete phase-matching so the POI coordinates are converted to projector POV
            projPOIs[i] = proj.PhaseMatch(poiCoords, xPhasemap, yPhasemap,
                xUnwrapper.stripeCount[-1], yUnwrapper.stripeCount[-1])

            if self.debug: self.__ShowPhasemaps(xPhasemap, yPhasemap)

        projReprojErrors, projRMSError = proj.Characterise(objectCoords, projPOIs)
        print(f"Projector reprojection error: {projRMSError}")

        if self.debug:
            projNormErrors = (projReprojErrors - np.min(projReprojErrors)) / np.ptp(projReprojErrors)
            for i in range(len(projPOIs)):
                self.__ShowPOIs(projImg.copy(), projPOIs[i], colourBy=projNormErrors[i])

        # And then.. use these individual characterisations to refine each other
        systemReprojErr = vision.RefineCharacterisations(cam.characterisation, proj.characterisation, objectCoords)
        print(f"Total reprojection error: {systemReprojErr}")

    # Private Functions

    def __RunAfterCaptureCallbacks(self, img, numImages):
        for func in self.m_CaptureCallbacks:
            func(img, numImages)

    def __ShowPOIs(self, img, poiCoords, colourBy=None):
        winName = f"Characterisation"

        if len(img.shape) == 2: img = image.ExpandN(img, 3)
        img = utils.ToNumpy(img)

        poiCoords = utils.ToNumpy(poiCoords)
        
        if colourBy is not None:
            # Sort by individual reprojection errors
            poiCoords = poiCoords[np.argsort(colourBy)]
            poiCoords = poiCoords.astype(np.uint16)

            for i in range(len(poiCoords)):
                # colour = (0.0, 1.0, 0.0) if reprojErrs[i] < 0 else (0.0, 0.0, 1.0)
                colour = (0.0, 1.0 - colourBy[i], float(colourBy[i]))
                img = cv2.circle(img, poiCoords[i], 1, colour, 2)
        
        else:
            for i, (x, y) in enumerate(poiCoords):
                img = cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                img = cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

        image.Show(img, winName)

    def __ShowPhasemaps(self, xPhasemap, yPhasemap):
        winName = f"Characterisation"

        xPhasemap = utils.ToNumpy(xPhasemap)
        yPhasemap = utils.ToNumpy(yPhasemap)

        ShowPhasemap(xPhasemap, winName)
        ShowPhasemap(yPhasemap, winName)