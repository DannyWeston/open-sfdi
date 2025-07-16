import pytest
import cProfile
import pstats

from pathlib import Path

from opensfdi import services, cloud, calibration as calib, reconstruction as recon
from opensfdi.devices import board, camera, projector
from opensfdi.phase import unwrap, shift
from opensfdi.utils import ProcessingContext, AlwaysNumpy

CUDA_LAUNCH_BLOCKING=1

from . import utils

expRoot = utils.DATA_ROOT / "realdata"

expRoot = Path("C:\\Users\\Dan\\Desktop\\realdata")

@pytest.mark.skip(reason="Not ready")
def test_calibration():
    # Fringe projection & calibration parameters

    # Use GPU for calculationsc v
    with ProcessingContext.UseGPU(True):

        proj = utils.FakeFPProjector(projector.ProjectorConfig(
            resolution=(1140, 912), channels=1,
            throwRatio=1.4, pixelSize=1.25)
        )

        imgRepo = services.FileImageRepo(expRoot, useExt='bmp')

        camRes = (3648, 5472)
        cam = camera.FileCamera(camera.CameraConfig(camRes, channels=1), images=[imgRepo.Get(f"calibration{i}") for i in range(462)])

        calibBoard = board.CircleBoard(circleSpacing=(7.778174593052023, 7.778174593052023), poiCount=(4, 13), inverted=True, staggered=True)
        # calibBoard = board.CircleBoard(poiCount=(4, 13), inverted=True, staggered=True)
        calibBoard.debug = False

        shifter = shift.NStepPhaseShift([6, 6, 9], mask=0.03)

        unwrapper = unwrap.MultiFreqPhaseUnwrap(
            numStripesVertical =    [1.0, 10.133333333333, 101.33333333],
            numStripesHorizontal =  [1.0, 7.9166666666666, 63.333333333]
        )

        # Profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        calibrator = calib.StereoCharacteriser(calibBoard)
        calibrator.Characterise(cam, proj, shifter, unwrapper, poseCount=11)
        
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(10)

    # Save the experiment information and the calibrated camera / projector
    # TODO: Implement services for camera and projector to make interface easier
    services.FileCameraConfigRepo(expRoot).Add(cam.config, "camera")
    services.FileProjectorRepo(expRoot).Add(proj.config, "projector")

    visionRepo = services.FileVisionConfigRepo(expRoot)
    visionRepo.Add(cam.visionConfig, "camera_vision")
    visionRepo.Add(proj.visionConfig, "projector_vision")

# @pytest.mark.skip(reason="Not ready")
def test_measurement():

    with ProcessingContext.UseGPU(True):
        # calibBoard = board.CircleBoard(circleSpacing=(0.03, 0.03), poiCount=(4, 13), inverted=True, staggered=True)
        reconstructor = recon.StereoReconstructor()

        camRepo = services.FileCameraConfigRepo(expRoot)
        projRepo = services.FileProjectorRepo(expRoot)
        visionRepo = services.FileVisionConfigRepo(expRoot)
        
        cam = camera.FileCamera(config=camRepo.Get("camera"), visionConfig=visionRepo.Get("camera_vision"))
        proj = utils.FakeFPProjector(config=projRepo.Get("projector"), visionConfig=visionRepo.Get("projector_vision"))

        imageRepo = services.FileImageRepo(expRoot, useExt='bmp')

        shifter = shift.NStepPhaseShift([6, 6, 9], mask=0.1)

        unwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 7.9166666666666, 63.33333333333333])

        cam.images = [imageRepo.Get(f"measurement{i}") for i in range(21)]

        measurementCloud, _ = reconstructor.Reconstruct(cam, proj, shifter, unwrapper, vertical=False)

    measurementCloud = AlwaysNumpy(measurementCloud)

    # Align and save
    # measurementCloud = cloud.AlignToCalibBoard(measurementCloud, cam, calibBoard)
    cloud.SaveNumpyAsCloud(expRoot / f"measurement.ply", measurementCloud)

    # Convert to open3d cloud
    measurementCloud = cloud.NumpyToCloud(measurementCloud)

        # # Load ground truth cloud
        # groundTruthMesh = cloud.LoadMesh(utils.DATA_ROOT / f"{obj}.stl")
        # groundTruthCloud = cloud.MeshToCloud(groundTruthMesh)

    cloud.DrawCloud(measurementCloud)