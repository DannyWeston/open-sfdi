import pytest

from opensfdi import services, cloud, calibration as calib, reconstruction as recon
from opensfdi.devices import board, camera, projector
from opensfdi.phase import unwrap, shift
from opensfdi.utils import ProcessingContext, AlwaysNumpy

from . import utils, profiling

expRoot = utils.DATA_ROOT / "realdata"

# @pytest.mark.skip(reason="Not ready")
def test_calibration():
    with ProcessingContext.UseGPU(True):
        proj = utils.FakeFPProjector(projector.ProjectorConfig(
            resolution=(1140, 912), channels=1,
            throwRatio=1.4, pixelSize=1.25)
        )
        # proj.debug = True

        imgRepo = services.FileImageRepo(expRoot, useExt='bmp')

        poseCount = 13

        cam = camera.FileCamera(
            camera.CameraConfig((3648, 5472), channels=1), 
            images=[imgRepo.Get(f"calibration{i}") for i in range(poseCount * 21 * 2)]
        )

        # TODO: Add some magic to determine orientation of board.
        # Currently the board has to be placed with the "sharp" corners formed by the convex hull
        #   at the top
        spacing = 0.007778174593052023
        calibBoard = board.CircleBoard(circleSpacing=(spacing, spacing), poiCount=(4, 13), inverted=True, staggered=True, poiMask=0.1,
                                       areaHint=(2000, 10000))
        # calibBoard.debug = True

        shifter = shift.NStepPhaseShift([6, 6, 9], mask=0.1)

        unwrapper = unwrap.MultiFreqPhaseUnwrap(
            numStripesVertical =    [1.0, 912.0/90.0, 912.0/9.0],
            numStripesHorizontal =  [1.0, 1140.0/144.0, 1140/18.0]
        )

        # Profiling
        # with profiling.ProfileCode():
        calibrator = calib.StereoCharacteriser(calibBoard)
        # calibrator.debug = True
        
        calibrator.Characterise(cam, proj, shifter, unwrapper, poseCount=poseCount)

    # Save the experiment information and the calibrated camera / projector
    # TODO: Implement services for camera and projector to make interface easier
    services.FileCameraConfigRepo(expRoot).Add(cam.config, "camera")
    services.FileProjectorRepo(expRoot).Add(proj.config, "projector")

    visionRepo = services.FileVisionConfigRepo(expRoot)
    visionRepo.Add(cam.visionConfig, "camera_vision")
    visionRepo.Add(proj.visionConfig, "projector_vision")

@pytest.mark.skip(reason="Not ready")
def test_measurement():
    # Fringe projection & calibration parameters

    with ProcessingContext.UseGPU(True):
        camRepo = services.FileCameraConfigRepo(expRoot)
        projRepo = services.FileProjectorRepo(expRoot)
        visionRepo = services.FileVisionConfigRepo(expRoot)

        imageRepo = services.FileImageRepo(expRoot, useExt='bmp')
        
        cam = camera.FileCamera(config=camRepo.Get("camera"), visionConfig=visionRepo.Get("camera_vision"))
        cam.images = [imageRepo.Get(f"measurement{i}") for i in range(21)]

        proj = utils.FakeFPProjector(config=projRepo.Get("projector"), visionConfig=visionRepo.Get("projector_vision"))

        shifter = shift.NStepPhaseShift([6, 6, 9], mask=0.1)
        unwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 1140.0/144.0, 1140/18.0])

        reconstructor = recon.StereoReconstructor()
        measurementCloud, dcImage, _ = reconstructor.Reconstruct(cam, proj, shifter, unwrapper, vertical=False)

    measurementCloud = AlwaysNumpy(measurementCloud)
    dcImage = AlwaysNumpy(dcImage)

    # Align and save
    # measurementCloud = cloud.AlignToCalibBoard(measurementCloud, cam, calibBoard)
    cloud.SaveArrayAsCloud(expRoot / f"measurement.ply", measurementCloud)

    # Convert to open3d cloud
    measurementCloud = cloud.ArrayToCloud(measurementCloud, dcImage)

        # # Load ground truth cloud
        # groundTruthMesh = cloud.LoadMesh(utils.DATA_ROOT / f"{obj}.stl")
        # groundTruthCloud = cloud.MeshToCloud(groundTruthMesh)

    cloud.DrawCloud(measurementCloud)