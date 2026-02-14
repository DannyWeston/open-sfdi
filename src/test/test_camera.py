import pytest
import numpy as np
import cv2

from pathlib import Path

from opensfdi import phase, devices, image, utils, services, cloud, colour, fringes, characterisation as ch

from .stub import StubProjector

exp_root = Path("D:\\results\\camera")

resolutions = [
    # (640, 360),
    # (960, 540),
    # (1200, 675),
    # (1440, 810),
    # (1920, 1080),
    # (2560, 1440),
    (3840, 2160),
]

objects = [
    # "Pillars",
    "PillarsB",
    # "Recess",
    # "SteppedPyramid"
]

@pytest.mark.skip(reason="Not ready")
def test_gamma():
    with utils.ProcessingContext.UseGPU(False): # Does not need to run on GPU
        xp = utils.ProcessingContext().xp

        intensity_count = 50

        for (w, h) in resolutions:
            test_root = exp_root / f"{w}x{h}"

            # Camera
            # camera = FileCamera(resolution=(400, 400), channels=1, refresh_rate=30.0)
            img_repo = services.FileImageRepo(test_root, use_ext='tif')

            # Calirbate Gamma
            intensities = xp.linspace(0.0, 1.0, intensity_count, dtype=xp.float32)
            gamma_corrector = colour.GammaCalibrator()

            # Gather the calibration images using the camera and projector
            gamma_imgs = [img_repo.Get(f"gamma{str(i).zfill(2)}") for i in range(intensity_count)]

            measurements = xp.asarray([gamma_corrector.ObtainValue(img, crop=(0.25, 0.0, 0.75, 0.5)) for img in gamma_imgs])
            gamma_corrector = gamma_corrector.Calculate(intensities, measurements)
            corrected = gamma_corrector.apply(measurements)

            assert xp.allclose(intensities, corrected, rtol=0.0, atol=0.005) # 0.5% tolerance

            gamma_repo = services.JSONRepository(test_root, overwrite=True)
            gamma_repo.Add(gamma_corrector, "gamma_corrector")

            print(f"Gamma calibration finished for {w}x{h}")

@pytest.mark.skip(reason="Not ready")
def test_calibration():
    with utils.ProcessingContext.UseGPU(True):
        xp = utils.ProcessingContext().xp

        # Parameters
        num_calib_imgs = 14

        # Create a projector
        projector = StubProjector(
            resolution=(1920, 1080), channels=1, refresh_rate=30.0, 
            throw_ratio=1.0, aspect_ratio=1.0, 
            char=ch.ZhangChar(
                focal_length=(5.0, 5.0),
                sensor_size=(6.0, 3.375),
                optical_centre=(0.5, 0.5),
                distort_mat=np.zeros((5, 1))
            )
        )

        # Phase shifters / unwrappers
        shifter_x = phase.NStepPhaseShift([12, 12, 12])
        shifter_y = phase.NStepPhaseShift([12, 12, 12])
        unwrapper_x = phase.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])
        unwrapper_y = phase.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

        for (w, h) in resolutions:
            print(f"Completing {w}x{h}")
            exp_dir = exp_root / f"{w}x{h}"
            img_repo = services.FileImageRepo(exp_dir, use_ext='tif')
            gamma_repo = services.JSONRepository[colour.GammaCorrector](exp_dir)

            # Camera (with images for FileCamera)
            camera = devices.FileCamera(resolution=(w, h), channels=1, refresh_rate=30.0, 
                images=list(img_repo.GetBy(f"calibration*", sorted=True)),
                char=ch.ZhangChar(
                    focal_length=(3.6, 3.6),
                    sensor_size=(3.76, 2.115),
                    optical_centre=(0.5, 0.5),
                )
            )
            
            # Characterisation board
            calib_board = ch.CircleBoard(
                poi_count=(13, 9), spacing=(14.6666666666, 13.359375),
                inverted=True, staggered=True, area_max=(w*h)//750
            )

            # Gamma calibration
            gamma_corrector = gamma_repo.Get("gamma_corrector")

            # Stereo Fringe Projection
            stereo_fpp = fringes.StereoFringeProjection()

            phasemap_xs = []
            phasemap_ys = []
            pois_cam = []

            fringe_rot = 0.0

            # AC/DC masking
            dc_mask = (1.0/4.0, 1.0)
            ac_mask = (0.2, 1.0)

            # Preallocate memory for images to be deposited into
            # Not required, but we are using a lot of uncompressed images
            imgs_x = xp.empty(shape=(sum(shifter_x.phase_counts), *camera.shape), dtype=np.float32)
            imgs_y = xp.empty_like(imgs_x)

            for i in range(num_calib_imgs):
                # Gather x-phasemap
                stereo_fpp.gather_imgs(
                    camera, projector, shifter_x.phase_counts, unwrapper_x.stripe_count, 
                    fringe_rot, reverse=False, gamma_corrector=gamma_corrector, 
                    out=imgs_x
                )

                phasemap_x, ac_img, dc_img = stereo_fpp.calculate_phasemap(imgs_x, shifter_x, unwrapper_x)

                # Identify pois on the DC img
                pois = calib_board.find_pois(dc_img)

                if pois is None:
                    print(f"Could not identify POIs, skipping collection {i+1}")
                    image.show_img(dc_img)
                    continue

                # As x-phasemap had pois successfully identified, gather y-phasemap
                stereo_fpp.gather_imgs(
                    camera, projector, shifter_y.phase_counts, unwrapper_y.stripe_count, 
                    fringe_rot + (np.pi / 2.0), reverse=True, gamma_corrector=gamma_corrector, 
                    out=imgs_y
                )

                phasemap_y, _, _ = stereo_fpp.calculate_phasemap(imgs_y, shifter_y, unwrapper_x)

                # Apply AC/DC filtering to phasemaps
                if dc_mask:
                    mask = image.ThresholdMask(dc_img, dc_mask[0], dc_mask[1])

                    dc_img[~mask] = xp.nan
                    phasemap_x[~mask] = xp.nan
                    phasemap_y[~mask] = xp.nan
                if ac_mask:
                    mask = image.ThresholdMask(ac_img, ac_mask[0], ac_mask[1])

                    dc_img[~mask] = xp.nan
                    phasemap_x[~mask] = xp.nan
                    phasemap_y[~mask] = xp.nan

                # Store calculated values
                pois_cam.append(pois)
                phasemap_xs.append(phasemap_x)
                phasemap_ys.append(phasemap_y)

            pois_cam = xp.asarray(pois_cam)

            # Calibrate the camera
            camera.char.execute(calib_board, pois_cam, camera.resolution)
            print(camera)

            # The camera is characterised, so we can now undistort the characterisation images/phasemaps
            pois_cam    = xp.asarray([camera.char.undistort_points(pois) for pois in pois_cam])
            phasemap_xs = [camera.char.undistort_img(phasemap_x) for phasemap_x in phasemap_xs]
            phasemap_ys = [camera.char.undistort_img(phasemap_y) for phasemap_y in phasemap_ys]

            # Convert cam pois phase values to projector coordinates
            pois_proj = xp.empty_like(pois_cam)
            for i, (pois, phasemap_x, phasemap_y) in enumerate(zip(pois_cam, phasemap_xs, phasemap_ys)):
                pois_proj[i, :, 0] = fringes.phase_to_coord(projector.resolution, pois, phasemap_x, unwrapper_x.stripe_count[-1], True, bilinear=True)
                pois_proj[i, :, 1] = fringes.phase_to_coord(projector.resolution, pois, phasemap_y, unwrapper_y.stripe_count[-1], False, bilinear=True)

            # Now calibrate the projector
            projector.char.execute(
                calib_board, pois_proj, projector.resolution,
                extraFlags=cv2.CALIB_FIX_PRINCIPAL_POINT
            )
            print(projector)
            
            # Jointly calibrate the system
            joint_char = camera.char.joint_char(projector.char, calib_board)
            print(joint_char)

            # Save the experiment information and the calibrated camera / projector
            cam_repo = services.JSONRepository[devices.FileCamera](exp_dir)
            cam_repo.Add(camera, "camera")

            proj_repo = services.JSONRepository[StubProjector](exp_dir)
            proj_repo.Add(projector, "projector")

            joint_repo = services.JSONRepository[ch.ZhangJointChar](exp_dir)
            joint_repo.Add(joint_char, "joint_char")
            print()

# @pytest.mark.skip(reason="Not ready")
def test_measurement():
    with utils.ProcessingContext.UseGPU(True):
        xp = utils.ProcessingContext().xp

        shifter = phase.NStepPhaseShift([12, 12, 12])
        unwrapper = phase.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

        dc_mask = (1.0/3.0, 1.0)
        ac_mask = (0.2, 1.0)

        for (w, h) in resolutions:
            test_root = exp_root / f"{w}x{h}"
            img_repo = services.FileImageRepo(test_root, use_ext='tif')

            camera_repo = services.JSONRepository[devices.FileCamera](test_root)
            camera = camera_repo.Get("camera")

            proj_repo = services.JSONRepository[StubProjector](test_root)
            projector = proj_repo.Get("projector")

            gamma_repo = services.JSONRepository[colour.GammaCorrector](test_root)
            gamma_corrector = gamma_repo.Get("gamma_corrector")

            # Create a reconstructor object
            stereo_fpp = fringes.StereoFringeProjection()

            for obj_name in objects:
                camera.images = list(img_repo.GetBy(obj_name, sorted=True))

                # Gather the fringe projection images
                imgs = stereo_fpp.gather_imgs(
                    camera, projector, shifter.phase_counts, unwrapper.stripe_count, 
                    0.0, reverse=True, gamma_corrector=gamma_corrector
                )

                # Convert the images to a phasemap and constant DC image
                phasemap, dc_img, ac_img = stereo_fpp.calculate_phasemap(imgs, shifter, unwrapper)

                # Check if we need to mask the shifted fringes using a DC image and thresholds
                if dc_mask:
                    mask = image.ThresholdMask(dc_img, dc_mask[0], dc_mask[1])
                    phasemap[~mask] = xp.nan

                if ac_mask:
                    mask = image.ThresholdMask(ac_img, ac_mask[0], ac_mask[1])
                    phasemap[~mask] = xp.nan

                # Reconstruct the point cloud
                np_cloud = stereo_fpp.reconstruct(phasemap, camera, projector, unwrapper.stripe_count[-1], use_x=False)

                # Filter out any values such that z < -25mm or z > +50mm from the origin 
                z_filter = cloud.filter_np_cloud(np_cloud, z=(-25, 50))

                np_cloud[~z_filter] = xp.nan
                dc_img[~z_filter] = xp.nan

                # Save and draw
                point_cloud = cloud.np_to_cloud(np_cloud, texture=dc_img / xp.nanmax(dc_img))
                cloud.show_cloud(point_cloud)