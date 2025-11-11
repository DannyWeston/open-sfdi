import pytest
import numpy as np

from opensfdi import services, cloud, calibration as calib, reconstruction as recon
from opensfdi.devices import board, camera, vision
from opensfdi.phase import unwrap, shift

from opensfdi.utils import ProcessingContext, ToNumpy, ProfileCode

from . import utils

expRoot = utils.DATA_ROOT / "realdata2"

# @pytest.mark.skip(reason="Not ready")
def test_calibration():
    with ProcessingContext.UseGPU(True):
        imgRepo = services.FileImageRepo(expRoot, useExt='bmp')

        poseCount = 21
        phases = [6, 6, 9]

        # Camera  
        cam = camera.FileCamera((3648, 5472), channels=1, refreshRate=30.0,
            character=vision.Characterisation( # Basler ACE
                sensorSizeGuess=(13.13, 8.76),
                focalLengthGuess=(16.0, 16.0),
                opticalCentreGuess=(0.5, 0.5)
            ),
            images=[imgRepo.Get(f"calibration{i}") for i in range(poseCount * sum(phases) * 2)]
        )

        # Projector
        proj = utils.FakeFPProjector(resolution=(912, 1140), channels=1, refreshRate=30.0,
            throwRatio=1.4, aspectRatio=1.25,
            character=vision.Characterisation( # DLP4500
                sensorSizeGuess=(9.855, 6.1614), 
                focalLengthGuess=(14.9212, 14.2160),
                opticalCentreGuess=(0.5, 1.0)
            ),
        )
        
        # Characterisation board
        spacing = 7.778174593052023
        calibBoard = board.CircleBoard(
            circleSpacing=(spacing, spacing), poiCount=(4, 13), inverted=True, staggered=True,
            poiMask=(0.15, 0.9), areaHint=(2000, 100000)
        )

        # Phase shifters and unwrappers
        shifter = shift.NStepPhaseShift(phases, (0.08, 0.9))
        xUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 912.0/90.0, 912.0/9.0])   # Phase change perpendicular to baseline (x for this scenario)
        yUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 1140.0/144.0, 1140/18.0]) # Phase change follows baseline (y for this scenario)    

        # Calibrator
        calibrator = calib.StereoCharacteriser(calibBoard)
        # calibrator.debug = True
        calibrator.Characterise(cam, proj, shifter, shifter, xUnwrapper, yUnwrapper, poseCount=poseCount)

    # Save the experiment information and the calibrated camera / projector
    # TODO: Implement services for camera and projector to make interface easier
    services.FileCameraRepo(expRoot).Add(cam, "camera")
    services.FileProjectorRepo(expRoot).Add(proj, "projector")

    visionRepo = services.FileVisionConfigRepo(expRoot)
    visionRepo.Add(cam.characterisation, "camera_vision")
    visionRepo.Add(proj.characterisation, "projector_vision")

@pytest.mark.skip(reason="Not ready")
def test_measurement():
    # Fringe projection & calibration parameters

    phases = [6, 6, 9]

    with ProcessingContext.UseGPU(True):
        camRepo = services.FileCameraRepo(expRoot)
        projRepo = services.FileProjectorRepo(expRoot)
        visionRepo = services.FileVisionConfigRepo(expRoot)

        imageRepo = services.FileImageRepo(expRoot, useExt='bmp')
        
        cam = camera.FileCamera(config=camRepo.Get("camera"), character=visionRepo.Get("camera_vision"))
        cam.images = [imageRepo.Get(f"measurement{i}") for i in range(sum(phases))]

        proj = utils.FakeFPProjector(config=projRepo.Get("projector"), visionConfig=visionRepo.Get("projector_vision"))

        shifter = shift.NStepPhaseShift(phases, mask=0.1)
        unwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 1140.0/144.0, 1140/18.0])

        reconstructor = recon.StereoReconstructor()
        measurementCloud, dcImage, _ = reconstructor.Reconstruct(cam, proj, shifter, unwrapper, vertical=False)

    measurementCloud = ToNumpy(measurementCloud)
    dcImage = ToNumpy(dcImage)

    # Align and save
    # measurementCloud = cloud.AlignToCalibBoard(measurementCloud, cam, calibBoard)
    cloud.SaveArrayAsCloud(expRoot / f"measurement.ply", measurementCloud)

    # Convert to open3d cloud
    measurementCloud = cloud.ArrayToCloud(measurementCloud, dcImage)

        # # Load ground truth cloud
        # groundTruthMesh = cloud.LoadMesh(utils.DATA_ROOT / f"{obj}.stl")
        # groundTruthCloud = cloud.MeshToCloud(groundTruthMesh)

    cloud.DrawCloud(measurementCloud)