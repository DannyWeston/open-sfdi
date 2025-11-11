import pytest
import numpy as np

from opensfdi import services, cloud, calibration as calib, reconstruction as recon
from opensfdi.devices import board, camera, projector, vision
from opensfdi.phase import unwrap, shift
from opensfdi.utils import ProcessingContext

from . import utils

expRoot = utils.DATA_ROOT / "projector"

resolutions = [
    # (3840, 2160),
    # (2560, 1440),
    (1920, 1080),
    # (1600, 900),
    # (1280, 720),
    # (1024, 768)
    # (960, 540),
    # (800, 600)
    # (640, 360)
]

objects = [
    "Pillars",
    "Recess",
    "SteppedPyramid"
]

# @pytest.mark.skip(reason="Not ready")
def test_calibration():
    with ProcessingContext.UseGPU(True):
        phases = [8, 8, 8]

        # Camera
        cam = camera.FileCamera((1080, 1920), channels=1, refreshRate=30.0,
            character = vision.Characterisation( # Basler ACE
                focalLengthGuess=(3.6, 3.6),
                sensorSizeGuess=(3.76, 2.115),
                opticalCentreGuess=(0.5, 0.5)
            )
        )

        # Phase manipulation
        shifter = shift.NStepPhaseShift(phases, contrastMask=(0.1, 0.8))  
        xUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 10.0, 100.0]) # Phase change perpendicular to baseline (x for this scenario)
        yUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 10.0, 100.0]) # Phase change follows baseline (y for this scenario)  

        # Characterisation board
        spacing = 0.007778174593052023 * 3
        calibBoard = board.CircleBoard(
            circleSpacing=(spacing, spacing),
            inverted=True, staggered=True,
            poiCount=(4, 13), poiMask=(0.1, 0.8),
            areaHint=(100, 10000)
        )

        calibrator = calib.StereoCharacteriser(calibBoard)
        def OnCapture(_, numCaptures):
            calibrator.m_GatheringImages = (numCaptures < 15)

        calibrator.AddCaptureCallback(OnCapture)
        calibrator.debug = True

        for (w, h) in resolutions:
            proj = utils.FakeFPProjector(resolution=(h, w), channels=1, refreshRate=30.0,
                throwRatio=1.0, aspectRatio=1.0,
                character = vision.Characterisation(
                    focalLengthGuess=(1.0, 1.0),
                    sensorSizeGuess=(16.0/9.0, 1.0),
                    opticalCentreGuess=(0.5, 0.5),
                    distortMat=np.zeros(5, dtype=np.float32)
                ),
            )

            expDir = expRoot / f"{w}x{h}"
            imgRepo = services.FileImageRepo(expDir, useExt='tif')

            cam.images = list(imgRepo.GetBy(f"calibration*", sorted=True))

            print(f"Characterising for {w}x{h}")
            calibrator.Characterise(cam, proj, shifter, shifter, xUnwrapper, yUnwrapper)
            print(f"Finished {w}x{h} characterisation")

            # Save the experiment information and the calibrated camera / projector
            services.FileCameraRepo(expDir).Add(cam, "camera")
            services.FileProjectorRepo(expDir).Add(proj, "projector")

            visionRepo = services.FileVisionConfigRepo(expDir)
            visionRepo.Add(cam.characterisation, "camera_vision")
            visionRepo.Add(proj.characterisation, "projector_vision")

@pytest.mark.skip(reason="Not ready")
def test_measurement():
    for (h, w) in resolutions[::-1]:
        testRoot = expRoot / f"{w}x{h}"

        with ProcessingContext.UseGPU(True):
            imageRepo = services.FileImageRepo(testRoot, useExt='tif')

            camRepo = services.FileCameraRepo(testRoot)
            projRepo = services.FileProjectorRepo(testRoot)
            visionRepo = services.FileVisionConfigRepo(testRoot)
            
            cam = camera.FileCamera(config=camRepo.Get("camera"), character=visionRepo.Get("camera_vision"))
            proj = utils.FakeFPProjector(config=projRepo.Get("projector"), visionConfig=visionRepo.Get("projector_vision"))

            reconstructor = recon.StereoReconstructor()
            shifter = shift.NStepPhaseShift([9, 9, 9], mask=0.1)
            unwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

            for obj in objects:
                cam.images = list(imageRepo.GetBy(obj, sorted=True))

                measurementCloud, dcImage, validPoints = reconstructor.Reconstruct(cam, proj, shifter, unwrapper, vertical=False)
                measurementCloud[:, 1] *= -1

                # Save and draw
                cloud.SaveArrayAsCloud(testRoot / f"{obj}.ply", measurementCloud)
                cloud.DrawCloud(cloud.ArrayToCloud(measurementCloud, colours=dcImage))