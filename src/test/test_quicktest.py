import pytest

from opensfdi import services, cloud, calibration as calib, reconstruction as recon
from opensfdi.devices import board, camera, projector
from opensfdi.phase import unwrap, shift

from . import utils

expRoot = utils.DATA_ROOT / "quicktest2"

@pytest.mark.skip(reason="Not ready")
def test_calibration():
    # Instantiate classes for calibration
    # Load the images to use in the calibration process

    # Fringe projection & calibration parameters
    shifterMask = 0.03
    phaseCount = 8
    stripeCounts = [1.0, 8.0, 64.0]
    boardPoses = 13

    proj = utils.FakeFPProjector(projector.ProjectorConfig(
        resolution=(912, 1140), channels=1,
        throwRatio=1.4, pixelSize=1.25)
    )

    imgRepo = services.FileImageRepo(expRoot, fileExt='.tif')

    camRes = (1080, 1920)
    cam = camera.FileCamera(camera.CameraConfig(camRes, channels=1), images=list(imgRepo.GetBy("calibration", sorted=True)))

    minArea = camRes[0] * camRes[1]
    maxArea = minArea * 4.5
    calibBoard = board.CircleBoard(circleSpacing=(0.03, 0.03), poiCount=(4, 13), inverted=True, staggered=True)
    # calibBoard.debug = True

    shifter = shift.NStepPhaseShift(phaseCount, mask=shifterMask)
    unwrapper = unwrap.MultiFreqPhaseUnwrap(stripeCounts)

    calibrator = calib.StereoCalibrator(calibBoard)
    calibrator.Calibrate(cam, proj, shifter, unwrapper, imageCount=boardPoses)

    # Save the experiment information and the calibrated camera / projector
    # TODO: Implement services for camera and projector to make interface easier
    services.FileCameraConfigRepo(expRoot, overwrite=True).Add(cam.config, "camera")
    services.FileProjectorRepo(expRoot, overwrite=True).Add(proj.config, "projector")

    services.FileVisionConfigRepo(expRoot, overwrite=True).Add(cam.visionConfig, "camera_vision")
    services.FileVisionConfigRepo(expRoot, overwrite=True).Add(proj.visionConfig, "projector_vision")

# @pytest.mark.skip(reason="Not ready")
def test_measurement():
    calibBoard = board.CircleBoard(circleSpacing=(0.03, 0.03), poiCount=(4, 13), inverted=True, staggered=True)

    phaseCount = 8
    stripeCounts = [1.0, 8.0, 64.0]

    objects = [
        "Icosphere",
        "Monkey",
        "Donut",
    ]

    reconstructor = recon.StereoReconstructor()

    camRepo = services.FileCameraConfigRepo(expRoot)
    projRepo = services.FileProjectorRepo(expRoot)
    visionRepo = services.FileVisionConfigRepo(expRoot)
    
    cam = camera.FileCamera(config=camRepo.Get("camera"), visionConfig=visionRepo.Get("camera_vision"))
    proj = utils.FakeFPProjector(config=projRepo.Get("projector"), visionConfig=visionRepo.Get("projector_vision"))

    imageRepo = services.FileImageRepo(expRoot, fileExt='.tif', channels=cam.config.channels)

    shifter = shift.NStepPhaseShift(phaseCount, 0.1)
    unwrapper = unwrap.MultiFreqPhaseUnwrap(stripeCounts)

    for obj in objects:
        cam.images = list(imageRepo.GetBy(obj, sorted=True))

        measurementCloud, _ = reconstructor.Reconstruct(cam, proj, shifter, unwrapper, stripeCounts[-1], useX=False)

        # Align and save
        # measurementCloud = cloud.AlignToCalibBoard(measurementCloud, cam, calibBoard)
        cloud.SaveNumpyAsCloud(expRoot / f"{obj}.ply", measurementCloud)

        # Convert to open3d cloud
        measurementCloud = cloud.NumpyToCloud(measurementCloud)

        # # Load ground truth cloud
        # groundTruthMesh = cloud.LoadMesh(utils.DATA_ROOT / f"{obj}.stl")
        # groundTruthCloud = cloud.MeshToCloud(groundTruthMesh)

        cloud.DrawCloud(measurementCloud)