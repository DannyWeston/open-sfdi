import pytest
import cv2

from pathlib import Path

from opensfdi import characterisation, stereo as calib, utils, services, cloud, colour, reconstruction as recon
from opensfdi.devices import FileCamera
from opensfdi.image import show_img
from opensfdi.phase import unwrap, shift

from .stub import StubProjector

exp_root = Path("D:\\results\\realdata3")

objects = [
    "Pillars",
    "Recess",
    "SteppedPyramid"
]

# @pytest.mark.skip(reason="Not ready")
def test_gamma():
    with utils.ProcessingContext.UseGPU(False): # Does not need to run on GPU
        xp = utils.ProcessingContext().xp

        intensity_count = 52

        # Camera
        # camera = FileCamera(resolution=(400, 400), channels=1, refresh_rate=30.0)
        img_repo = services.FileImageRepo(exp_root / "gamma", use_ext='bmp')

        # Projector
        # projector = FakeFPProjector(resolution=(1920, 1080), channels=1, refresh_rate=30.0, throw_ratio=1.0, aspect_ratio=1.0)

        # Calirbate Gamma
        intensities = xp.linspace(0.0, 1.0, intensity_count, dtype=xp.float32)
        gamma_calibrator = colour.GammaCalibrator()

        # Gather the calibration images using the camera and projector
        # gamma_imgs = gamma_corrector.GatherImages(camera, projector, intensities)

        gamma_imgs = [img_repo.Get(f"Img_{i}") for i in range(intensity_count)]

        measurements = xp.asarray([gamma_calibrator.ObtainValue(img, crop=(0.25, 0.0, 0.75, 0.5)) for img in gamma_imgs])
        gamma_calibrator.Plot(intensities, measurements)

        gamma_calibrator = gamma_calibrator.Calculate(intensities, measurements)
        corrected = gamma_calibrator.apply(measurements)

        # print(f"Before: {np.abs(intensities - measurements).max() * 100 :.2f}%")
        # print(f"After: {np.abs(intensities - corrected).max() * 100 :.2f}%")

        assert xp.allclose(intensities, corrected, rtol=0.0, atol=0.005) # 0.5% tolerance

        gamma_repo = services.JSONRepository(exp_root / "gamma", overwrite=True)
        gamma_repo.Add(gamma_calibrator, "gamma_corrector")

@pytest.mark.skip(reason="Not ready")
def test_calibration():
    utils.ProcessingContext.UseGPU(False)

    img_repo = services.FileImageRepo(exp_root, use_ext='bmp')

    # Parameters
    num_calib_imgs = 20
    contrast_mask = (0.0784, 0.90196)
    
    # Camera - Basler Ace
    cam_w, cam_h = (5472, 3648)
    camera = FileCamera(resolution=(cam_w, cam_h), channels=1, refresh_rate=30.0,
        images=[img_repo.Get(f"calibration{i}") for i in range(num_calib_imgs * 21 * 2)]
    )
    camera_char = characterisation.ZhangChar(
        sensor_size=(13.1328, 8.7552),
        focal_length=(16.0, 16.0),
        optical_centre=(0.5, 0.5)
    )

    # DLP4500
    proj_w, proj_h = (912, 1140)
    projector = StubProjector(resolution=(proj_w, proj_h), channels=1, refresh_rate=30.0,
        throw_ratio=1.0, aspect_ratio=1.0,
    )
    projector_char = characterisation.ZhangChar(
        # sensor_size=(9.855, 6.1614*2.0),
        # focal_length=(23.64, 23.64),
        # optical_centre=(0.5, 1.0)
    )

    # Characterisation board
    calib_board = characterisation.CircleBoard(
        poi_count=(13, 4), spacing=(7.778174593052023/2, 7.778174593052023),
        inverted=True, staggered=True, area_max=(cam_w*cam_h)//2000
    )

    # Corrector for image gamma
    gamma_repo = services.JSONRepository(exp_root / "gamma")
    gamma_corrector = gamma_repo.Get("gamma_corrector")
    gamma_corrector = None

    camera_pois = []
    proj_pois = []
    dc_imgs = []

    fringeRot = 0.0

    # Gather all the images and calculate phasemaps
    # Can use GPU for this!
    with utils.ProcessingContext.UseGPU(True):
        xp = utils.ProcessingContext().xp

        # Phase shifters / unwrappers (GPU)
        xShifter = shift.NStepPhaseShift([6, 6, 9])
        yShifter = shift.NStepPhaseShift([6, 6, 9])

        xUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, proj_w/90.0, proj_w/9.0])
        yUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, proj_h/144.0, proj_h/18.0])

        # Stereo calibrator
        calibrator = calib.ZhangCharacteriser(calib_board, gamma_corrector, contrast_mask)

        # Debug img for projector
        proj_debug_img = xp.zeros(projector.shape, dtype=xp.float32)

        for i in range(num_calib_imgs):
            print(f"Current image set: {i+1}")

            phiX, dc_img = calibrator.Measure(camera, projector, xShifter, xUnwrapper, fringeRot, reverse=True)

            pois = calib_board.find_pois(dc_img)

            # Get y phasemap as successful
            phiY, _ = calibrator.Measure(camera, projector, yShifter, yUnwrapper, fringeRot + (xp.pi / 2.0), reverse=True)

            if pois is None:
                print(f"Could not identify POIs, skipping collection {i+1}")
                show_img(dc_img, size=(1920, 1080))
                continue

            # Debugging camera POI view
            # characterisation.ShowPOIs(dc_img, calib_board.poi_count, pois, size=(1920, 1080))

            # Store detected camera POI coordinates
            camera_pois.append(pois)
            dc_imgs.append(dc_img)

            # Convert to projector coordinates
            pois2 = xp.empty_like(pois)

            pois2[:, 0] = projector.PhaseToCoord(pois, phiX, xUnwrapper.stripeCount[-1], True, bilinear=True)
            pois2[:, 1] = projector.PhaseToCoord(pois, phiY, yUnwrapper.stripeCount[-1], False, bilinear=True)

            # Debugging projector POI view
            # characterisation.ShowPOIs(proj_debug_img, calib_board.poi_count, pois2)

            proj_pois.append(pois2)

        camera_pois = xp.asarray(camera_pois)
        dc_imgs = xp.asarray(dc_imgs)
        proj_pois = xp.asarray(proj_pois)

    # Calibrate the camera
    camReprojErrs, camRMSErr = camera_char.execute(calib_board, camera_pois, camera.resolution)
    print(f"Camera R_error: {camRMSErr:.2f} (std: {xp.std(camReprojErrs):.2f})")

    # Calibrate the projector
    projReprojErrs, projRMSErr = projector_char.execute(
        calib_board, proj_pois, projector.resolution,
        extraFlags=cv2.CALIB_FIX_PRINCIPAL_POINT
    )
    print(f"Projector R_error: {projRMSErr:.2f} (std: {xp.std(projReprojErrs):.2f})")
    
    # Jointly calibrate the system
    # jointReprojErrs, jointRMSErr = camera_char.JointCalibration(projector_char, calibBoard)
    # print(f"Joint R_error: {jointRMSErr}")

    # Save the experiment information and the calibrated camera / projector
    camRepo = services.JSONRepository[FileCamera](exp_root)
    camRepo.Add(camera, "camera")

    projRepo = services.JSONRepository[StubProjector](exp_root)
    projRepo.Add(projector, "projector")

    calibRepo = services.JSONRepository[characterisation.ZhangChar](exp_root)
    calibRepo.Add(camera_char, "camera_vision")
    calibRepo.Add(projector_char, "projector_vision")

@pytest.mark.skip(reason="Not ready")
def test_measurement():
    utils.ProcessingContext.UseGPU(False)

    # Corrector for image gamma
    gamma_repo = services.JSONRepository[colour.GammaCorrector](exp_root / "gamma")
    gamma_corrector = gamma_repo.Get("gamma_corrector")

    img_repo = services.FileImageRepo(exp_root, use_ext='tif')
    camera_repo = services.JSONRepository[FileCamera](exp_root)
    proj_repo = services.JSONRepository[StubProjector](exp_root)
    calib_repo = services.JSONRepository[characterisation.ZhangChar](exp_root)

    # Camera & Projector
    camera: FileCamera = camera_repo.Get("camera")
    camera_char = calib_repo.Get("camera_vision")

    projector: StubProjector = proj_repo.Get("projector")
    projector_char = calib_repo.Get("projector_vision")

    with utils.ProcessingContext.UseGPU(True):
        unwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

        # Create a reconstructor object
        shifter = shift.NStepPhaseShift([6, 6, 9])
        reconstructor = recon.StereoFringeProjection(gamma_corrector, contrast_mask=(1.0/3.0, 1.0))

        for obj_name in objects:
            camera.images = list(img_repo.GetBy(obj_name, sorted=True))

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