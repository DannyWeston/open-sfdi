import pytest

from opensfdi import services, cloud, calibration as calib, reconstruction as recon
from opensfdi.devices import board, camera, projector
from opensfdi.phase import unwrap, shift
from opensfdi.utils import ProcessingContext, ToNumpy

from . import utils

expRoot = utils.DATA_ROOT / "phase"

# phaseCounts = list(range(6, 10))
phaseCounts = [5]

objects = [
    "Pillars",
    "Recess",
    "SteppedPyramid"
]

# @pytest.mark.skip(reason="Not ready")
def test_calibration():
    for p in phaseCounts:
        print(f"Characterising with {p} phase count")

        testRoot = expRoot / str(p)

        with ProcessingContext.UseGPU(True):
            proj = utils.FakeFPProjector(projector.ProjectorConfig(
                resolution=(1080, 1920), channels=1,
                throwRatio=3.0/2.0, aspectRatio=4.0/3.0)
            )

            imgRepo = services.FileImageRepo(testRoot, useExt='tif')
            images = list(imgRepo.GetBy(f"calibration", sorted=True))

            spacing = 0.007778174593052023 * 3.0
            cam = camera.FileCamera(camera.CameraConfig((1080, 1920), channels=1), images=images)   
            calibBoard = board.CircleBoard(circleSpacing=(spacing, spacing), poiCount=(4, 13), inverted=True, staggered=True, poiMask=0.1, areaHint=(1000, 10000))

            shifter = shift.NStepPhaseShift([p, p, p], mask=0.1)
            unwrapper = unwrap.MultiFreqPhaseUnwrap(
                numStripesVertical =    [1.0, 8.0, 64.0],
                numStripesHorizontal =  [1.0, 8.0, 64.0]
            )

            calibrator = calib.StereoCharacteriser(calibBoard)
            calibrator.Characterise(cam, proj, shifter, unwrapper, poseCount=16)

        services.FileCameraRepo(testRoot).Add(cam.config, "camera")
        services.FileProjectorRepo(testRoot).Add(proj.config, "projector")

        visionRepo = services.FileVisionConfigRepo(testRoot)
        visionRepo.Add(cam.characterisation, "camera_vision")
        visionRepo.Add(proj.characterisation, "projector_vision")

@pytest.mark.skip(reason="Not ready")
def test_measurement():
    for p in phaseCounts:
        testRoot = expRoot / str(p)

        with ProcessingContext.UseGPU(True):
            imageRepo = services.FileImageRepo(testRoot, useExt='tif')

            camRepo = services.FileCameraRepo(testRoot)
            projRepo = services.FileProjectorRepo(testRoot)
            visionRepo = services.FileVisionConfigRepo(testRoot)
            
            cam = camera.FileCamera(config=camRepo.Get("camera"), character=visionRepo.Get("camera_vision"))
            proj = utils.FakeFPProjector(config=projRepo.Get("projector"), visionConfig=visionRepo.Get("projector_vision"))

            reconstructor = recon.StereoReconstructor()
            shifter = shift.NStepPhaseShift([p, p, p], mask=0.2)
            unwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

            for obj in objects:
                cam.images = list(imageRepo.GetBy(obj, sorted=True))

                measurementCloud, dcImage, validPoints = reconstructor.Reconstruct(cam, proj, shifter, unwrapper, vertical=False)
                measurementCloud[:, 1] *= -1

                measurementCloud = ToNumpy(measurementCloud)
                dcImage = ToNumpy(dcImage)

                # Save and draw
                cloud.SaveArrayAsCloud(testRoot / f"{obj}.ply", measurementCloud)
                cloud.DrawCloud(cloud.ArrayToCloud(measurementCloud, dcImage))