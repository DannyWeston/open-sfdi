import numpy as np
import time

from abc import ABC, abstractmethod
from numpy.polynomial import polynomial as P

from .image import DC, Show
from .phase import unwrap, shift, ShowPhasemap
from .devices import camera, projector, board, vision, FringeProject

from .utils import ProcessingContext, AlwaysNumpy

class FPCharacteriser(ABC):
    @abstractmethod
    def __init__(self):
        # TODO: Allow for other calibration artefacts
        pass

class StereoCharacteriser(FPCharacteriser):
    def __init__(self, calibBoard: board.CalibrationBoard):
        super().__init__()

        self.m_CalibBoard = calibBoard

    def GatherPhasemap(self, camera: camera.Camera, projector: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, vertical=True):
        xp = ProcessingContext().xp
        
        # TODO: Add some flag to save the images whilst gathering?
        shifted = xp.empty((len(unwrapper.stripeCount), *camera.config.shape), dtype=xp.float32)

        print(shifted.dtype is xp.float32)

        dcImage = None

        unwrapper.vertical = vertical
        shifter.vertical = vertical

        for i, (numStripes, phaseCount) in enumerate(zip(unwrapper.stripeCount, shifter.phaseCounts)):

            imgs = xp.empty(shape=(phaseCount, *camera.config.resolution), dtype=xp.float32)

            phases = (xp.arange(phaseCount) / phaseCount) * 2.0 * xp.pi

            # TODO: Vectorise
            for j in range(phaseCount):
                imgs[j] = xp.asarray(FringeProject(camera, projector, numStripes, phases[j]).rawData)

            if i == 0: dcImage = DC(imgs)
            
            shifted[i] = shifter.Shift(imgs)
            # ShowPhasemap(shifted[i], name=f"Wrapped phasemap {i}", size=(1000, 600))

        # Calculate unwrapped phase maps
        return unwrapper.Unwrap(shifted), dcImage

    def Characterise(self, cam: camera.Camera, proj: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, poseCount=15):
        startTime = time.time()

        phasemaps = []
        cameraCoords = []
        objectCoords = []

        for i in range(poseCount):

            tempTime = time.time()

            # Vertical phase maps
            proj.stripeRotation = 0.0
            vertPhasemap, dcImage = self.GatherPhasemap(cam, proj, shifter, unwrapper, vertical=True)
            # ShowPhasemap(vertPhasemap, name=f"Unwrapped phasemap {i} (vertical)", size=(1000, 600))
            # Show(dcImage, name=f"DC Image {i}", size=(1000, 600))

            xp = ProcessingContext().xp

            corners = self.m_CalibBoard.FindPOIS(dcImage)

            if corners is None:
                print(f"Could not find checkerboard corners for image {i}")
                continue

            # Horizontal phase maps (ignore DC image as we don't need it)
            proj.stripeRotation = np.pi / 2.0
            horiPhasemap, _ = self.GatherPhasemap(cam, proj, shifter, unwrapper, vertical=False)
            # ShowPhasemap(horiPhasemap, name=f"Unwrapped phasemap {i} (horizontal)", size=(1000, 600))

            print(f"Identified corners and calculated phasemaps {i}")
            print(f"Total time: {time.time() - tempTime}")

            # Corners successfully detected, register them for characterisation
            phasemaps.append(vertPhasemap)
            phasemaps.append(horiPhasemap)
            cameraCoords.append(corners)
        
        # Never big enough to warrant GPU usage for creation 
        # And cv2 needs np type of data...
        cameraCoords = np.asarray(cameraCoords)
        objectCoords = np.repeat(self.m_CalibBoard.GetPOICoords()[np.newaxis, ...], len(cameraCoords), axis=0)

        # Debug message
        # TODO: Change to logger
        print(f"{len(cameraCoords)} images with POIs correctly identified")

        # Characterise the camera individually
        cam.Characterise(objectCoords, cameraCoords)
        print(f"Camera reprojection error: {cam.visionConfig.reprojErr}")

        # Characterise the projector individually
        proj.Characterise(objectCoords, cameraCoords, phasemaps, unwrapper.stripeCount[-1])
        print(f"Projector reprojection error: {proj.visionConfig.reprojErr}")

        # Refine the characterisations using each other
        cam.visionConfig, proj.visionConfig, reprojErr = vision.RefineCharacterisations(cam.visionConfig, proj.visionConfig, objectCoords)
        print(f"Total reprojection error: {reprojErr}")

        # self.m_Metadata = {
        #     "PoseCount"         : poseCount,
        #     "BoardsDetected"    : len(cameraCoords),

        #     "TimeElapsed"       : 0.0,
        #     "ReprojErr"         : reprojErr
        # }

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

def checkerboard_centre(cb_size, square_size):
    cx = (cb_size[0] / 2 - 1) * square_size
    cy = (cb_size[1] / 2 - 1) * square_size 

    return cx, cy