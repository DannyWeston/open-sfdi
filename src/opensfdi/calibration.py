import cv2
import numpy as np
import time


from abc import ABC, abstractmethod
from numpy.polynomial import polynomial as P

from .image import Show, DC
from .phase import unwrap, shift
from .devices import camera, projector, board, vision

from . import reconstruction as recon

class FPCalibrator(ABC):
    @abstractmethod
    def __init__(self):
        # TODO: Allow for other calibration artefacts
        pass

class StereoCalibrator(FPCalibrator):
    def __init__(self, calib_board: board.CalibrationBoard):
        super().__init__()

        self.m_CalibBoard = calib_board

    def GatherPhaseMap(self, camera: camera.Camera, projector: projector.FringeProjector, shifter: shift.PhaseShift, phi_unwrap: unwrap.PhaseUnwrap):
        # TODO: Add some flag to save the images whilst gathering?
        sfs = phi_unwrap.GetFringeCount()
        phases = shifter.GetPhases()

        shifted = np.empty((len(sfs), *camera.config.shape), dtype=np.float32)
        
        dcImage = None

        # Calculate the wrapped phase maps
        for j, sf in enumerate(sfs):
            imgs = FringeProject(camera, projector, sf, phases)

            if j == 0: dcImage = DC(imgs)

            shifted[j] = shifter.shift(imgs)

        # Calculate unwrapped phase maps
        return phi_unwrap.unwrap(shifted), dcImage
    
    def GatherDC(self, camera: camera.Camera, projector: projector.FringeProjector):
        # Gather a single full intensity DC img
        return FringeProject(camera, projector, 0.0, [0.0])[0]

    def Calibrate(self, camera: camera.Camera, projector: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, imageCount=15):
        start_time = time.time()

        camShape = camera.config.shape

        phasemaps = np.empty(shape=(imageCount * 2, *camShape), dtype=np.float32) 
        dc_imgs = np.empty(shape=(imageCount, *camShape), dtype=np.float32)

        for i in range(imageCount):
            # Vertical phase maps
            projector.stripeRotation = 0.0
            phasemaps[2*i], dc_imgs[i] = self.GatherPhaseMap(camera, projector, shifter, unwrapper)
            # phase.show_phasemap(phasemaps[2*i])
            # show_image(dc_imgs[i])
            
            # Horizontal phase maps (ignore DC image as we don't need it)
            projector.stripeRotation = np.pi / 2.0
            phasemaps[2*i+1], _ = self.GatherPhaseMap(camera, projector, shifter, unwrapper)
            # phase.show_phasemap(phasemaps[2*i+1])

        # Calibrate the camera and projector
        self.__Calibrate(camera, projector, dc_imgs, phasemaps, numStripes=unwrapper.GetFringeCount()[-1])

        end_time = (time.time() - start_time) * 1000

        result = CalibrationResult()
        result.time_taken = end_time
        result.phi_shifter = shifter.__class__.__name__
        result.phi_unwrapper = unwrapper.__class__.__name__

        return result

    def __Calibrate(self, cam: camera.Camera, proj: projector.FringeProjector, boardImages, phasemaps: np.ndarray, numStripes) -> recon.StereoProfil:
        """ The triangular stereo calibration model for fringe projection setups. """

        N = len(boardImages)

        # Corner finding algorithm needs greyscale images
        assert N * 2 == len(phasemaps)

        worldCoords = []
        cameraCoords = []
        validPhasemaps = []

        boardCoords = self.m_CalibBoard.GetPOICoords()

        for i in range(N): # We can skip duplicate checkerboards as is bad practice
            corners = self.m_CalibBoard.FindPOIS(boardImages[i])

            if corners is None:
                print(f"Could not find checkerboard corners for image {i}")
                continue

            worldCoords.append(boardCoords)
            cameraCoords.append(corners)

            # Add the phasemaps
            validPhasemaps.append(phasemaps[2*i])
            validPhasemaps.append(phasemaps[2*i+1])

            # DEBUG: Draw corners detected
            # show_image(cv2.drawChessboardCorners(cb_imgs[i], self.__cb_size, corners, True))

        worldCoords = np.array(worldCoords)
        cameraCoords = np.array(cameraCoords)

        # Finished phase manipulation, now just need to calibrate camera and projector
        print(f"{len(cameraCoords)} images with POIs correctly identified")

        cam.config = cam.Calibrate(worldCoords, cameraCoords)
        proj.config = proj.Calibrate(worldCoords, cameraCoords, validPhasemaps, numStripes)

        print(f"Camera reprojection error: {cam.config.visionConfig.reprojErr}")
        print(f"Projector reprojection error: {proj.config.visionConfig.reprojErr}")

        reprojErr = vision.RefineDevices(cam.config, proj.config, worldCoords)
        print(f"Total reprojection error: {reprojErr}")

        # TODO: Use
        self.m_Metadata = {
            "BoardCount"        : N,
            "BoardsDetected"    : len(worldCoords),

            "TimeElapsed"       : 0.0,
            "ReprojErr"         : reprojErr
        } 

        return True

class IPhaseHeightCalibrator(ABC):
    @abstractmethod
    def calibrate(self, phasemaps, heights) -> recon.PhaseHeightReconstructor:
        raise NotImplementedError

class PolynomialProfilCalibrator(IPhaseHeightCalibrator):
    def __init__(self, degree=5):
        self.degree = degree

    def calibrate(self, phasemaps, heights) -> recon.PolynomialProfil:
        """
            The polynomial calibration model for fringe projection setups.

            Note:   
                - The moving plane must be parallel to the camera 
                - The first phasemap is taken to be the reference phasemap
        """
        # Calculate phase difference maps at each height
        # Phase difference between ref and h = 0 is zero
        _, h, w = phasemaps.shape

        ph_maps = np.empty_like(phasemaps)
        ph_maps[0] = 0.0                            # Phase difference between baseline and baseline is 0.0
        ph_maps[1:] = phasemaps[1:] - phasemaps[0]  # Phase difference between baseline and height-increments

        # Polynomial fit on a pixel-by-pixel basis to its height value
        polydata = np.empty(shape=(self.degree + 1, h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                polydata[:, y, x] = P.polyfit(ph_maps[:, y, x], heights, deg=self.degree)

        return PolynomialProfil(polydata)

class CalibrationResult:
    def __init__(self):
        self.time_taken = None
        self.phi_unwrapper = None
        self.phi_shifter = None

        self.number_of_calibration_images = None

class MeasurementResult:
    def __init__(self):
        self.time_taken = None

# Utility functions

def checkerboard_centre(cb_size, square_size):
    cx = (cb_size[0] / 2 - 1) * square_size
    cy = (cb_size[1] / 2 - 1) * square_size 

    return cx, cy

def FringeProject(camera: camera.Camera, projector: projector.FringeProjector, sf, phases) -> np.ndarray:
    projector.numStripes = sf
    N = len(phases)

    imgs = np.empty(shape=(N, *camera.config.shape))

    for i in range(N):
        projector.phase = phases[i]
        projector.Display()
        imgs[i] = camera.Capture().raw_data
        # imgs[i] = add_gaussian(imgs[i], sigma=0.1)

    return imgs