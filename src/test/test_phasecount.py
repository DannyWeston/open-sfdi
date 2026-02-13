import pytest
import numpy as np
import cv2

from pathlib import Path

from opensfdi import characterisation, stereo as calib, utils, services, cloud, colour, reconstruction as recon
from opensfdi.devices import FileCamera
from opensfdi.image import show_img
from opensfdi.phase import unwrap, shift

from .stub import StubProjector

exp_root = Path("D:\\results\\phase_count")

phase_counts = [
    # 3,
    # 4,
    # 5,
    # 6,
    # 7,
    # 8,
    # 9,
    # 10,
    # 11,
    # 12,
]

objects = [
    "Pillars",
    "Recess",
    "SteppedPyramid"
]

@pytest.mark.skip(reason="Not ready")
def test_gamma():
    with utils.ProcessingContext.UseGPU(False): # Does not need to run on GPU
        xp = utils.ProcessingContext().xp

        intensity_count = 50

        test_root = exp_root / "gamma"

        # Camera
        img_repo = services.FileImageRepo(test_root, use_ext='tif')

        # Calirbate Gamma
        intensities = xp.linspace(0.0, 1.0, intensity_count, dtype=xp.float32)
        gamma_corrector = colour.GammaCalibrator()

        # Gather the calibration images using the camera and projector
        # gamma_imgs = gamma_corrector.GatherImages(camera, projector, intensities)

        gamma_imgs = [img_repo.Get(f"gamma{str(i).zfill(2)}") for i in range(intensity_count)]

        measurements = xp.asarray([gamma_corrector.ObtainValue(img, crop=(0.25, 0.0, 0.75, 0.5)) for img in gamma_imgs])
        gamma_corrector = gamma_corrector.Calculate(intensities, measurements)
        corrected = gamma_corrector.apply(measurements)

        assert xp.allclose(intensities, corrected, rtol=0.0, atol=0.005) # 0.5% tolerance

        gamma_repo = services.JSONRepository(test_root, overwrite=True)
        gamma_repo.Add(gamma_corrector, "gamma_corrector")

# @pytest.mark.skip(reason="Not ready")
def test_calibration():
    utils.ProcessingContext.UseGPU(False)

    # Parameters
    num_calib_imgs = 14

    # Camera
    w, h = 1920, 1080
    camera = FileCamera(resolution=(w, h), channels=1, refresh_rate=30.0)
    camera_char = characterisation.ZhangChar(
        focal_length=(3.6, 3.6),
        sensor_size=(3.76, 2.115),
        optical_centre=(0.5, 0.5)
    )

    # Projector
    projector = StubProjector(
        resolution=(1920, 1080), channels=1, refresh_rate=30.0, 
        throw_ratio=1.0, aspect_ratio=1.0
    )
    projector_char = characterisation.ZhangChar(
        focal_length=(5.0, 5.0),
        sensor_size=(6.0, 3.375),
        optical_centre=(0.5, 0.5),
    )

    # Characterisation board
    calib_board = characterisation.CircleBoard(
        poi_count=(13, 9), spacing=(14.6666666666, 13.359375),
        inverted=True, staggered=True, area_max=w*h//500
    )
    
    # Corrector for image gamma
    gamma_repo = services.JSONRepository(exp_root / "gamma")
    gamma_corrector = gamma_repo.Get("gamma_corrector")
    
    with utils.ProcessingContext.UseGPU(True):
        # Calibration objects
        xUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])
        yUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

        # Fringe projection calibrator
        calibrator = calib.ZhangCharacteriser(calib_board, gamma_corrector, contrast_mask=(1.0/3.0, 1.0))

        for phase_count in phase_counts:
            exp_dir = exp_root / str(phase_count)

            # Phase shifters / unwrappers
            xShifter = shift.NStepPhaseShift([phase_count, phase_count, phase_count])
            yShifter = shift.NStepPhaseShift([phase_count, phase_count, phase_count])

            # Set camera images
            img_repo = services.FileImageRepo(exp_dir, use_ext='tif')
            camera.images = list(img_repo.GetBy(f"calibration*", sorted=True))

            # Begin
            camera_pois = []
            proj_pois = []
            dc_imgs = []

            fringeRot = 0.0

            with utils.ProcessingContext.UseGPU(True):
                xp = utils.ProcessingContext().xp

                for i in range(num_calib_imgs):
                    print(f"Current image set: {i+1}")

                    phiX, dc_img = calibrator.Measure(camera, projector, xShifter, xUnwrapper, fringeRot)

                    pois = calib_board.find_pois(dc_img)

                    # Get y phasemap as successful
                    phiY, _ = calibrator.Measure(camera, projector, yShifter, yUnwrapper, fringeRot + (np.pi / 2.0), reverse=True)

                    if pois is None:
                        print(f"Could not identify POIs, skipping collection {i+1}")
                        show_img(dc_img)
                        continue

                    # Store detected camera POI coordinates
                    camera_pois.append(pois)
                    dc_imgs.append(dc_img)

                    # Convert to projector coordinates
                    pois2 = xp.empty_like(pois)

                    pois2[:, 0] = projector.PhaseToCoord(pois, phiX, xUnwrapper.stripeCount[-1], True, bilinear=True)
                    pois2[:, 1] = projector.PhaseToCoord(pois, phiY, yUnwrapper.stripeCount[-1], False, bilinear=True)
                    
                    proj_pois.append(pois2)

                    # Debugging projector POI view
                    # characterisation.ShowPOIs(proj_debug_img, calib_board.poi_count, pois2)

                camera_pois = xp.asarray(camera_pois)
                proj_pois = xp.asarray(proj_pois)
                dc_imgs = xp.asarray(dc_imgs)

                # Calibrate the camera
                camReprojErrs, camRMSErr = camera_char.execute(calib_board, camera_pois, camera.resolution)
                print(f"Camera R_error: {camRMSErr:.2f} (std: {xp.std(camReprojErrs):.2f})")

                # Now calibrate the projector
                projReprojErrs, projRMSErr = projector_char.execute(
                    calib_board, proj_pois, projector.resolution,
                    extraFlags=cv2.CALIB_FIX_PRINCIPAL_POINT
                )
                print(f"Projector R_error: {projRMSErr:.2f} (std: {xp.std(projReprojErrs):.2f})")
                
                # Jointly calibrate the system
                # jointReprojErrs, jointRMSErr = camera_char.JointCalibration(projector_char, calibBoard)
                # print(f"Joint R_error: {jointRMSErr}")

                # Save the experiment information and the calibrated camera / projector
                camRepo = services.JSONRepository[FileCamera](exp_dir)
                camRepo.Add(camera, "camera")

                projRepo = services.JSONRepository[StubProjector](exp_dir)
                projRepo.Add(projector, "projector")

                calibRepo = services.JSONRepository[characterisation.ZhangChar](exp_dir)
                calibRepo.Add(camera_char, "camera_vision")
                calibRepo.Add(projector_char, "projector_vision")

# @pytest.mark.skip(reason="Not ready")
def test_measurement():
    utils.ProcessingContext.UseGPU(False)

    # Corrector for image gamma
    gamma_repo = services.JSONRepository[colour.GammaCorrector](exp_root / "gamma")
    gamma_corrector = gamma_repo.Get("gamma_corrector")

    with utils.ProcessingContext.UseGPU(True):
        unwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

        for phase_count in phase_counts:
            test_root = exp_root / str(phase_count)

            img_repo = services.FileImageRepo(test_root, use_ext='tif')
            camera_repo = services.JSONRepository[FileCamera](test_root)
            proj_repo = services.JSONRepository[StubProjector](test_root)
            calib_repo = services.JSONRepository[characterisation.ZhangChar](test_root)

            # Camera & Projector
            camera: FileCamera = camera_repo.Get("camera")
            camera_char = calib_repo.Get("camera_vision")

            projector: StubProjector = proj_repo.Get("projector")
            projector_char = calib_repo.Get("projector_vision")

            # Create a reconstructor object
            shifter = shift.NStepPhaseShift([phase_count, phase_count, phase_count])
            reconstructor = recon.StereoFringeProjection(gamma_corrector, contrast_mask=(1.0/3.0, 1.0))

            for obj_name in objects:
                camera.images = list(img_repo.GetBy(obj_name, sorted=True))

                point_cloud, texture, valid_points = reconstructor.reconstruct(
                    camera, projector, camera_char, projector_char,
                    shifter, unwrapper, use_x=False, reverse=True
                )

                # Remove values that exceed the measureable volume
                point_cloud, z_filter = cloud.FilterCloud(point_cloud, z=(-25, 50))
                texture = texture[z_filter]

                # Save and draw
                cloud.SaveArrayAsCloud(test_root / f"{obj_name}.ply", point_cloud)
                cloud.DrawCloud(cloud.ArrayToCloud(point_cloud, texture=texture))