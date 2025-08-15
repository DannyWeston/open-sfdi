import pytest

from opensfdi import services, cloud, calibration as calib, reconstruction as recon
from opensfdi.devices import board, camera, projector
from opensfdi.phase import unwrap, shift
from opensfdi.utils import ProcessingContext

from . import utils

expRoot = utils.DATA_ROOT / "projector"

resolutions = [
    (2160, 3840),
    (1440, 2560),
    (1080, 1920),
    (900, 1600),
    (720, 1280),
    (540, 960),
    (360, 640)
]

objects = [
    "Pillars",
    "Recess",
    "SpherePlinth",
    "SteppedPyramid"
]

# @pytest.mark.skip(reason="Not ready")
def test_calibration():
    for (h, w) in resolutions:
        print(f"Characterising for {w}x{h}")

        testRoot = expRoot / f"{w}x{h}"

        imgRepo = services.FileImageRepo(testRoot, useExt='tif')

        with ProcessingContext.UseGPU(True):
            
            proj = utils.FakeFPProjector(projector.ProjectorConfig(
                resolution=(h, w), channels=1,
                throwRatio=1.4, pixelSize=1.25)
            )

            cam = camera.FileCamera(camera.CameraConfig((1080, 1920), channels=3))
            cam.images = list(imgRepo.GetBy(f"calibration", sorted=True))

            calibBoard = board.CircleBoard(circleSpacing=(0.03, 0.03), poiCount=(4, 13), inverted=True, staggered=True, poiMask=0.1)

            shifter = shift.NStepPhaseShift([9, 9, 9], mask=0.1)
            unwrapper = unwrap.MultiFreqPhaseUnwrap(
                numStripesVertical =    [1.0, 8.0, 64.0],
                numStripesHorizontal =  [1.0, 8.0, 64.0]
            )

            calibrator = calib.StereoCharacteriser(calibBoard)
            calibrator.Characterise(cam, proj, shifter, unwrapper, poseCount=13)

        services.FileCameraConfigRepo(testRoot).Add(cam.config, "camera")
        services.FileProjectorRepo(testRoot).Add(proj.config, "projector")

        visionRepo = services.FileVisionConfigRepo(testRoot)
        visionRepo.Add(cam.visionConfig, "camera_vision")
        visionRepo.Add(proj.visionConfig, "projector_vision")

        print(f"Finished {w}x{h} characterisation")

@pytest.mark.skip(reason="Not ready")
def test_measurement():
    for (h, w) in resolutions[::-1]:
        testRoot = expRoot / f"{w}x{h}"

        with ProcessingContext.UseGPU(True):
            imageRepo = services.FileImageRepo(testRoot, useExt='tif')

            camRepo = services.FileCameraConfigRepo(testRoot)
            projRepo = services.FileProjectorRepo(testRoot)
            visionRepo = services.FileVisionConfigRepo(testRoot)
            
            cam = camera.FileCamera(config=camRepo.Get("camera"), visionConfig=visionRepo.Get("camera_vision"))
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