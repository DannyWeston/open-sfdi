import numpy as np
import time

from abc import ABC, abstractmethod

from .phase import unwrap, shift, ShowPhasemap
from .devices import camera, projector, board, vision, FringeProject
from .utils import ProcessingContext

from . import image

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

    @property
    def useChannel(self):
        return self.m_UseChannel
    
    @useChannel.setter
    def useChannel(self, value):
        self.m_UseChannel = value

    def GatherPhasemap(self, camera: camera.Camera, projector: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, vertical=True):
        xp = ProcessingContext().xp
        
        # TODO: Add some flag to save the images whilst gathering?
        shifted = xp.empty((len(unwrapper.stripeCount), *camera.config.shape), dtype=xp.float32)

        unwrapper.vertical = vertical
        shifter.vertical = vertical

        dcImage = None

        for i, (numStripes, N) in enumerate(zip(unwrapper.stripeCount, shifter.phaseCounts)):

            imgs = xp.empty(shape=(N, *camera.config.shape), dtype=xp.float32)

            phases = (xp.arange(N) * 2.0 * xp.pi) / N

            # TODO: Vectorise
            for j in range(N):
                rawData = FringeProject(camera, projector, numStripes, phases[j]).rawData

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
        return unwrapper.Unwrap(shifted), dcImage

    def Characterise(self, cam: camera.Camera, proj: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, poseCount=15):
        startTime = time.time()

        phasemaps = []
        cameraCoords = []
        objectCoords = []

        boardCoords = self.m_CalibBoard.GetPOICoords()

        for i in range(poseCount):
            tempTime = time.time()

            # Vertical phase maps
            proj.stripeRotation = 0.0
            vertPhasemap, dcImage = self.GatherPhasemap(cam, proj, shifter, unwrapper, vertical=True)

            if self.debug:
                ShowPhasemap(vertPhasemap)
                image.Show(dcImage, "DC Image")

            corners = self.m_CalibBoard.FindPOIS(image.ToGrey(dcImage))

            if corners is None:
                print(f"Could not find checkerboard corners for image {i}")
                continue

            # Horizontal phase maps (ignore DC image as we don't need it)
            proj.stripeRotation = np.pi / 2.0
            horiPhasemap, _ = self.GatherPhasemap(cam, proj, shifter, unwrapper, vertical=False)
            if self.debug: ShowPhasemap(horiPhasemap)

            print(f"Successfully identified corners for set {i} ({time.time() - tempTime:.3f}s)")

            # Corners successfully detected, register them for characterisation
            # Default to using first shifted fringes for now
            # TODO: Add for colour selection or averaging
            if vertPhasemap.ndim == 3:
                phasemaps.append(vertPhasemap[:, :, self.useChannel])
                phasemaps.append(horiPhasemap[:, :, self.useChannel])
            else:
                phasemaps.append(vertPhasemap)
                phasemaps.append(horiPhasemap)
        
            cameraCoords.append(corners)
        
        # Never big enough in memory to warrant GPU usage for creation
        # And cv2 needs np type of data...
        cameraCoords = np.asarray(cameraCoords)
        objectCoords = np.repeat(boardCoords[np.newaxis, ...], len(cameraCoords), axis=0)

        # TODO: Change to logger
        print(f"{len(cameraCoords)} images with POIs correctly identified")

        # Characterise the camera individually
        cam.Characterise(objectCoords, cameraCoords)
        print(f"Camera reprojection error: {cam.visionConfig.reprojErr}")

        # Characterise the projector individually
        proj.Characterise(objectCoords, cameraCoords, phasemaps, unwrapper.vertNumStripes[-1], unwrapper.horiNumStripes[-1])
        print(f"Projector reprojection error: {proj.visionConfig.reprojErr}")

        # # Refine the characterisations using each other
        # cam.visionConfig, proj.visionConfig, reprojErr = vision.RefineCharacterisations(
        #     cam.visionConfig, proj.visionConfig, objectCoords)
        # print(f"Total reprojection error: {reprojErr}")

        result = CalibrationResult()
        result.time_taken = (time.time() - startTime) * 1000
        result.phi_shifter = shifter.__class__.__name__
        result.phi_unwrapper = unwrapper.__class__.__name__

        return result

class CalibrationResult:
    def __init__(self):
        self.time_taken = None
        self.phi_unwrapper = None
        self.phi_shifter = None

        self.number_of_calibration_images = None

# Utility functions