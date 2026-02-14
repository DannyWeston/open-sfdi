import pytest
import numpy as np
import cv2

import matplotlib.pyplot as plt

from opensfdi import characterisation, services, cloud, colour, reconstruction as recon, stereo as calib
from opensfdi.devices import FileCamera
from opensfdi.image import show_img
from opensfdi.phase import show_phasemap, unwrap, shift
from opensfdi.utils import ProcessingContext
from opensfdi.fringes import sinusoidal_pattern

from . import stub

expRoot = stub.DATA_ROOT / "projector"

resolutions = [
    # (3840, 2160),
    # (2560, 1440),
    # (1920, 1080),
    (1600, 900),
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

@pytest.mark.skip(reason="Not ready")
def test_gamma():
    with ProcessingContext.UseGPU(False):
        xp = ProcessingContext().xp

        intensity_count = 50

        # Camera
        camera = FileCamera(resolution=(1920, 1080), channels=1, refresh_rate=30.0)
        imgRepo = services.FileImageRepo(expRoot / f"{camera.resolution[0]}x{camera.resolution[1]}", use_ext='tif')
        camera.images = list(imgRepo.GetBy("gamma*", sorted=True))

        # Projector
        projector = stub.StubProjector(resolution=(1920, 1080), channels=1, refresh_rate=30.0, throw_ratio=1.0, aspect_ratio=1.0)

        # Calirbate Gamma
        intensities = xp.linspace(0.0, 1.0, intensity_count, dtype=xp.float32)
        calib = colour.GammaCalibrator()

        # Gather the calibration images
        gamma_imgs = calib.GatherImages(camera, projector, intensities)

        measurements = calib.ObtainValues(gamma_imgs) # Mean values of cropped images
        corrector = calib.Calculate(intensities, measurements)
        corrected = corrector.apply(measurements)

        # print(f"Before: {np.abs(intensities - measurements).max() * 100 :.2f}%")
        # print(f"After: {np.abs(intensities - corrected).max() * 100 :.2f}%")

        assert xp.allclose(intensities, corrected, rtol=0.0, atol=0.005) # 0.5% tolerance

# @pytest.mark.skip(reason="Not ready")
def test_calibration():
    with ProcessingContext.UseGPU(True):
        xp = ProcessingContext().xp

        # Camera
        camera = FileCamera(resolution=(1920, 1080), channels=1, refresh_rate=30.0)
        camera_char = characterisation.ZhangChar(
            focal_length=(3.6, 3.6),
            sensor_size=(3.76, 2.115),
            optical_centre=(0.5, 0.5)
        )

        # Parameters
        numImages = 14

        # Phase manipulation
        xShifter = shift.NStepPhaseShift([12, 12, 12])
        yShifter = shift.NStepPhaseShift([12, 12, 12])
        xUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])
        yUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

        # Characterisation board
        poiCount = (13, 9)
        spacing = (14.6666666666, 13.359375)
        calibBoard = characterisation.CircleBoard(
            spacing=spacing, poiCount=poiCount,
            inverted=True, staggered=True,
            area_max=(500, 8000)
        )

        # Corrector for image gamma
        gamma_corrector = colour.GammaCorrector(
            a=0.9031509820825411,
            gamma=0.9996699423580444,
            b=0.000137525007885489
        )

        calibrator = calib.ZhangCharacteriser(calibBoard, gamma_corrector, contrast_mask=None)

        projector = stub.StubProjector(resolution=(1920, 1080), channels=1, refresh_rate=30.0, throw_ratio=1.0, aspect_ratio=1.0)
        projector_char = characterisation.ZhangChar(
            focal_length=(5.0, 5.0),
            sensor_size=(6.0, 3.375),
            optical_centre=(0.5, 0.5),
        )

        expDir = expRoot / "1920x1080"
        imgRepo = services.FileImageRepo(expDir, use_ext='tif')

        camera.images = list(imgRepo.GetBy(f"calibration*", sorted=True))

        phiXs = []
        phiYs = []
        camPOIs = []
        camImgs = []

        fringeRot = 0.0

        # Gather all the images
        for i in range(numImages):
            phiX, camImg = calibrator.Measure(camera, projector, xShifter, xUnwrapper, fringeRot)
            # ShowPhasemap(phiX)

            pois = calibBoard.find_pois(camImg)

            # Get y phasemap as successful
            phiY, _ = calibrator.Measure(camera, projector, yShifter, yUnwrapper, fringeRot + (np.pi / 2.0), reverse=True)
            # ShowPhasemap(phiY)

            if pois is None:
                print(f"Could not identify POIs, skipping collection {i+1}")
                show_img(camImg)
                continue

            camImgs.append(camImg)
            camPOIs.append(pois)
            phiXs.append(phiX)
            phiYs.append(phiY)

        camPOIs = xp.asarray(camPOIs)

        # Calibrate the camera
        camReprojErrs, camRMSErr = camera_char.execute(
            calibBoard, camPOIs, camera.resolution,
        )
        
        print(f"Camera R_error: {camRMSErr} (std: {xp.std(camReprojErrs)})")

        # The camera is characterised, so we can now undistort the calibration images/phasemaps
        camPOIs = xp.asarray([camera_char.undistort_points(pois) for pois in camPOIs])
        phiXs   = [camera_char.undistort_img(phiX) for phiX in phiXs]
        phiYs   = [camera_char.undistort_img(phiY) for phiY in phiYs]

        # Convert camPOI phase values to projector coordinates
        projPOIs = xp.empty_like(camPOIs)
        for i, (pois, phiX, phiY) in enumerate(zip(camPOIs, phiXs, phiYs)):
            projPOIs[i, :, 0] = projector.PhaseToCoord(pois, phiX, xUnwrapper.stripeCount[-1], True, bilinear=True)
            projPOIs[i, :, 1] = projector.PhaseToCoord(pois, phiY, yUnwrapper.stripeCount[-1], False, bilinear=True)

        # for pois in projPOIs:
        #     characterisation.ShowPOIs(xp.zeros(shape=(projector.shape)), poiCount, pois)

        # Now calibrate the projector
        projReprojErrs, projRMSErr = projector_char.execute(
            calibBoard, projPOIs, projector.resolution,
            extraFlags=cv2.CALIB_FIX_PRINCIPAL_POINT
        )
        print(f"Projector R_error: {projRMSErr} (std: {xp.std(projReprojErrs)})")
        
        # Jointly calibrate the system
        # jointReprojErrs, jointRMSErr = camera_char.JointCalibration(projector_char, calibBoard)
        # print(f"Joint R_error: {jointRMSErr}")

        # Save the experiment information and the calibrated camera / projector
        camRepo = services.JSONRepository[FileCamera](expDir)
        camRepo.Add(camera, "camera")

        projRepo = services.JSONRepository[stub.StubProjector](expDir)
        projRepo.Add(projector, "projector")

        calibRepo = services.JSONRepository[characterisation.ZhangChar](expDir)
        calibRepo.Add(camera_char, "camera_vision")
        calibRepo.Add(projector_char, "projector_vision")

        return

@pytest.mark.skip(reason="Not ready")
def test_measurement():
    testRoot = expRoot / "1920x1080"
    with ProcessingContext.UseGPU(True):
        imageRepo = services.FileImageRepo(testRoot, use_ext='tif')

        # Phase manipulation
        shifter = shift.NStepPhaseShift([12, 12, 12])
        unwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

        # JSON Repository for loading json config files
        camera_repo = services.JSONRepository[FileCamera](testRoot)
        proj_repo = services.JSONRepository[stub.StubProjector](testRoot)
        calib_repo = services.JSONRepository[characterisation.ZhangChar](testRoot)
        
        camera: FileCamera = camera_repo.Get("camera")
        camera_char = calib_repo.Get("camera_vision")

        projector: stub.StubProjector = proj_repo.Get("projector")
        projector_char = calib_repo.Get("projector_vision")

        # Corrector for image gamma
        gamma_corrector = colour.GammaCorrector(
            a=0.9031509820825411,
            gamma=0.9996699423580444,
            b=0.000137525007885489
        )

        # Create a reconstructor object
        reconstructor = recon.StereoFringeProjection(gamma_corrector, contrast_mask=(1.0/3.0, 1.0))

        camera.images = list(imageRepo.GetBy("Recess", sorted=True))

        measurementCloud, dcImage, validPoints = reconstructor.reconstruct(
            camera, projector, camera_char, projector_char,
            shifter, unwrapper, use_x=False, reverse=True
        )

        # measurementCloud[:, 1] *= -1 # Flip

        # Remove values that exceed the measureable volume
        measurementCloud, heightFilter = cloud.filter_np_cloud(measurementCloud, z=(-25, 50))
        dcImage = dcImage[heightFilter]

        # Save and draw
        # cloud.SaveArrayAsCloud(testRoot / f"{obj}.ply", measurementCloud)
        cloud.DrawCloud(cloud.np_to_cloud(measurementCloud, texture=dcImage))