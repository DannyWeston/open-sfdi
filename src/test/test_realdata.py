import pytest
import numpy as np

from pathlib import Path

from opensfdi import image, services, cloud, calibration as calib, reconstruction as recon
from opensfdi.devices import camera, characterisation
from opensfdi.phase import unwrap, shift, ShowPhasemap

from opensfdi.utils import ProcessingContext, ToNumpy, ProfileCode

from . import utils

expRoot = Path("D:\\metrology\\final\\test2\\formatted")


# @pytest.mark.skip(reason="Not ready")
def test_calibration():
    imgRepo = services.FileImageRepo(expRoot, useExt='bmp')

    with ProcessingContext.UseGPU(False):
        xp = ProcessingContext().xp

        # Camera - Basler Ace
        cam = camera.FileCamera((3648, 5472), channels=1, refreshRate=30.0,
            character = characterisation.Characterisation(
                sensorSizeGuess=(13.13, 8.76),
                focalLengthGuess=(16.0, 16.0),
                opticalCentreGuess=(0.5, 0.5)
            )
        )
        cam.images = [imgRepo.Get(f"calibration{i}") for i in range(1092)]

        # Projector - DLP4500
        proj = utils.FakeFPProjector(resolution=(1140, 912), channels=1, refreshRate=30.0,
            throwRatio=1.0, aspectRatio=1.0,
            character=characterisation.Characterisation(
                # sensorSizeGuess=(9.855, 6.1614*2.0),
                # focalLengthGuess=(23.64, 23.64),
                # opticalCentreGuess=(0.5, 1.0)
            ),
        )

        # Parameters
        contrastMask = (0.0784, 0.90196)
        numImages = 10

        # Characterisation board
        spacing = 7.778174593052023
        calibBoard = characterisation.CircleBoard(
            circleSpacing=(spacing, spacing), poiCount=(4, 13),
            inverted=True, staggered=True, areaHint=(100, 100000)
        )

        # Phase manipulation
        xShifter = shift.NStepPhaseShift([6, 6, 9])
        yShifter = shift.NStepPhaseShift([6, 6, 9])
        xUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, proj.resolution[1]/90.0, proj.resolution[1]/9.0])
        yUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, proj.resolution[0]/144.0, proj.resolution[0]/18.0])
        calibrator = calib.StereoCharacteriser(calibBoard)


        # Experiment
        phiXs = []
        phiYs = []
        camPOIs = []
        camImgs = []

        xFringeRot = 0.0

        # Gather all the images
        for i in range(numImages):
            print(i)

            phiX, camImg = calibrator.Measure(cam, proj, xShifter, xUnwrapper, xFringeRot)
            # Show(camImg, size=(1600, 900))
            # ShowPhasemap(phiX, size=(1600, 900))
            print(phiX.min(), phiX.max())

            pois = calibBoard.FindPOIS(camImg)
            # characterisation.ShowPOIs(camImg.copy(), pois, size=(1600, 900))

            # Get y phasemap as successful
            phiY, _ = calibrator.Measure(cam, proj, yShifter, yUnwrapper, xFringeRot + (np.pi / 2.0))
            # ShowPhasemap(phiY, size=(1600, 900))
            print(phiY.min(), phiY.max())

            if pois is None:
                print(f"Could not identify POIs, skipping collection {i+1}")
                image.Show(camImg, size=(1600, 900))
                continue

            camImgs.append(camImg)
            camPOIs.append(pois)
            phiXs.append(phiX)
            phiYs.append(phiY)

        camPOIs = xp.asarray(camPOIs)

        # Calibrate the camera
        camReprojErrs, camRMSErr = cam.Characterise(calibBoard, camPOIs)
        print(f"Camera RMS reprojection error: {camRMSErr}")
        print(f"Camera STD reprojection error: {xp.std(camReprojErrs)}")

        # for pois, errs, img in zip(camPOIs, camReprojErrs, camImgs): 
        #     characterisation.ShowPOIs(img.copy(), pois, size=(1600, 900), colourBy=errs)

        # The camera is characterised, so we can now undistort the calibration images/phasemaps
        # camImgs = [cam.characterisation.Undistort(img) for img in camImgs]
        # camPOIs = xp.asarray([cam.characterisation.UndistortPoints(pois) for pois in camPOIs])
        # phiXs   = [cam.characterisation.Undistort(phiX) for phiX in phiXs]
        # phiYs   = [cam.characterisation.Undistort(phiY) for phiY in phiYs]

        # Convert camPOI phase values to projector coordinates
        projPOIs = xp.asarray([
            proj.PhaseToCoord(pois, phiX, phiY, xUnwrapper.stripeCount[-1], yUnwrapper.stripeCount[-1], bilinear=True)
            for (pois, phiX, phiY) in zip(camPOIs, phiXs, phiYs)
        ])

        # Now calibrate the projector
        projReprojErrs, projRMSErr = proj.Characterise(calibBoard, projPOIs)
        print(f"Projector RMS reprojection error: {projRMSErr}")
        print(f"Projector STD reprojection error: {xp.std(projReprojErrs)}")

        for pois, errs in zip(projPOIs, projReprojErrs): 
            characterisation.ShowPOIs(xp.zeros(shape=(proj.shape)), pois, size=(1600, 900), colourBy=errs)

        reprojErr = characterisation.RefineCharacterisations(
            cam.characterisation, proj.characterisation, 
            calibBoard
        )
        print(f"System RMS reprojection error: {reprojErr}")

        # Save the experiment information and the calibrated camera / projector
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