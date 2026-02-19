import pytest
import numpy as np
import cv2

from opensfdi import characterisation, services, cloud, colour, stereo as calib
from opensfdi.devices import FileCamera
from opensfdi.image import show_img
from opensfdi.phase import unwrap, shift
from opensfdi.utils import ProcessingContext

from . import stub

exp_root = stub.DATA_ROOT / "mcmf"

objects = [
    "Pillars",
    "Recess",
    "SteppedPyramid"
]


# @pytest.mark.skip(reason="Not ready")
def test_gamma():
    with ProcessingContext.UseGPU(False): # Does not need to run on GPU
        xp = ProcessingContext().xp

        intensity_count = 50

        # Camera
        # camera = FileCamera(resolution=(400, 400), channels=1, refresh_rate=30.0)
        img_repo = services.FileImageRepo(exp_root, use_ext='tif')

        # Projector
        # projector = utils.FakeFPProjector(resolution=(1920, 1080), channels=1, refresh_rate=30.0, throw_ratio=1.0, aspect_ratio=1.0)

        # Calirbate Gamma
        intensities = xp.linspace(0.0, 1.0, intensity_count, dtype=xp.float32)
        gamma_corrector = colour.GammaCalibrator()

        # Gather the calibration images
        # gamma_imgs = gamma_corrector.GatherImages(camera, projector, intensities)

        measurements = gamma_corrector.ObtainValues(list(img_repo.GetBy("gamma*", sorted=True))) # Mean values of cropped images
        gamma_corrector = gamma_corrector.Calculate(intensities, measurements)
        corrected = gamma_corrector.apply(measurements)

        # print(f"Before: {np.abs(intensities - measurements).max() * 100 :.2f}%")
        # print(f"After: {np.abs(intensities - corrected).max() * 100 :.2f}%")

        assert xp.allclose(intensities, corrected, rtol=0.0, atol=0.005) # 0.5% tolerance

        gamma_repo = services.JSONRepository(exp_root, overwrite=True)
        gamma_repo.Add(gamma_corrector)

@pytest.mark.skip(reason="Not ready")
def test_calibration():
    with ProcessingContext.UseGPU(True):
        xp = ProcessingContext().xp

        # Parameters
        num_calib_imgs = 10

        # Camera
        camera = FileCamera(resolution=(400, 400), channels=1, refresh_rate=30.0)
        camera_char = characterisation.ZhangChar(
            focal_length=(1.2, 1.2),
            sensor_size=(1.0, 1.0),
            optical_centre=(0.5, 0.5)
        )

        # Create a projector
        projector = stub.StubProjector(
            resolution=(1920, 1080), channels=1, refresh_rate=30.0, 
            throw_ratio=1.0, aspect_ratio=1.0
        )
        projector_char = characterisation.ZhangChar(
            focal_length=(5.0, 5.0),
            sensor_size=(6.0, 3.375),
            optical_centre=(0.5, 0.5),
        )

        # Phase shifters / unwrappers
        xShifter = shift.NStepPhaseShift([6, 6, 6])
        yShifter = shift.NStepPhaseShift([6, 6, 6])
        xUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])
        yUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])
        
        # Characterisation board
        calib_board = characterisation.CircleBoard(
            spacing=(21.25, 19.921875), poiCount=(9, 9),
            inverted=True, staggered=True, area_max=(32, 2000)
        )

        exp_dir = exp_root / f"{400}x{400}"
        img_repo = services.FileImageRepo(exp_dir, use_ext='tif')

        # Corrector for image gamma
        gamma_corrector = colour.GammaCorrector(
            a=0.9031509820825411,
            gamma=0.9996699423580444,
            b=0.000137525007885489
        )

        calibrator = calib.ZhangCharacteriser(calib_board, gamma_corrector, contrast_mask=None)
        camera.images = list(img_repo.GetBy(f"calibration*", sorted=True))

        # Begin
        phiXs = []
        phiYs = []
        camPOIs = []
        camImgs = []

        fringeRot = 0.0

        # Gather all the images
        for i in range(num_calib_imgs):
            phiX, camImg = calibrator.Measure(camera, projector, xShifter, xUnwrapper, fringeRot)
            # ShowPhasemap(phiX)

            pois = calib_board.find_pois(camImg)
            characterisation.show_pois(camImg.copy(), calib_board._poi_count, pois)

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
            calib_board, camPOIs, camera.resolution,
        )
        
        print(f"Camera R_error: {camRMSErr:.2f} (std: {xp.std(camReprojErrs):.2f})")

        # The camera is characterised, so we can now undistort the calibration images/phasemaps
        camPOIs = xp.asarray([camera_char.undistort_points(pois) for pois in camPOIs])
        phiXs   = [camera_char.undistort_img(phiX) for phiX in phiXs]
        phiYs   = [camera_char.undistort_img(phiY) for phiY in phiYs]

        # Convert camPOI phase values to projector coordinates
        projPOIs = xp.empty_like(camPOIs)
        for i, (pois, phiX, phiY) in enumerate(zip(camPOIs, phiXs, phiYs)):
            projPOIs[i, :, 0] = projector.PhaseToCoord(pois, phiX, xUnwrapper.stripeCount[-1], True, bilinear=True)
            projPOIs[i, :, 1] = projector.PhaseToCoord(pois, phiY, yUnwrapper.stripeCount[-1], False, bilinear=True)

        # Now calibrate the projector
        projReprojErrs, projRMSErr = projector_char.execute(
            calib_board, projPOIs, projector.resolution,
            extraFlags=cv2.CALIB_FIX_PRINCIPAL_POINT
        )
        print(f"Projector R_error: {projRMSErr:.2f} (std: {xp.std(projReprojErrs):.2f})")
        
        # Jointly calibrate the system
        # jointReprojErrs, jointRMSErr = camera_char.JointCalibration(projector_char, calibBoard)
        # print(f"Joint R_error: {jointRMSErr}")

        # Save the experiment information and the calibrated camera / projector
        camRepo = services.JSONRepository[FileCamera](exp_dir)
        camRepo.Add(camera, "camera")

        projRepo = services.JSONRepository[stub.StubProjector](exp_dir)
        projRepo.Add(projector, "projector")

        calibRepo = services.JSONRepository[characterisation.ZhangChar](exp_dir)
        calibRepo.Add(camera_char, "camera_vision")
        calibRepo.Add(projector_char, "projector_vision")

@pytest.mark.skip(reason="Not ready")
def test_measurement():
    with ProcessingContext.UseGPU(True):

        # Phase manipulation
        shifter = shift.NStepPhaseShift([6, 6, 6])
        unwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

        # JSON Repository for loading json config files
        imageRepo = services.FileImageRepo(exp_root, use_ext='tif')
        camera_repo = services.JSONRepository[FileCamera](exp_root)
        proj_repo = services.JSONRepository[stub.StubProjector](exp_root)
        calib_repo = services.JSONRepository[characterisation.ZhangChar](exp_root)
        gamma_repo = services.JSONRepository[colour.GammaCorrector](exp_root)

        camera: FileCamera = camera_repo.Get("camera")
        camera_char = calib_repo.Get("camera_vision")

        projector: stub.StubProjector = proj_repo.Get("projector")
        projector_char = calib_repo.Get("projector_vision")

        # Corrector for image gamma
        # gamma_corrector = gamma_repo.Get("gamma_corrector")
        gamma_corrector = colour.GammaCorrector(
            a=0.9031509820825411,
            gamma=0.9996699423580444,
            b=0.000137525007885489
        )

        # Create a reconstructor object
        reconstructor = recon.StereoFringeProjection(gamma_corrector, contrast_mask=(1.0/3.0, 1.0))

        for obj_name in objects:
            camera.images = list(imageRepo.GetBy(obj_name, sorted=True))

            measurementCloud, dcImage, validPoints = reconstructor.reconstruct(
                camera, projector, camera_char, projector_char,
                shifter, unwrapper, use_x=False, reverse=True
            )

            # Remove values that exceed the measureable volume
            measurementCloud, heightFilter = cloud.filter_np_cloud(measurementCloud, z=(-25, 50))
            dcImage = dcImage[heightFilter]

            # Save and draw
            cloud.save_np_as_ply(exp_root / f"{obj_name}.ply", measurementCloud)
            cloud.DrawCloud(cloud.np_to_cloud(measurementCloud, texture=dcImage))