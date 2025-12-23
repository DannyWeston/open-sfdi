import pytest
import numpy as np

from opensfdi import services, cloud, calibration as calib, reconstruction as recon
from opensfdi.devices import camera, characterisation, projector
from opensfdi.image import Show
from opensfdi.phase import ShowPhasemap, unwrap, shift
from opensfdi.utils import ProcessingContext

from . import utils

expRoot = utils.DATA_ROOT / "projector"

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

def RegionBasedIntensities(imgs, regionsX=3, regionsY=3):
    xp = ProcessingContext().xp

    h, w, *_ = imgs[0].shape

    totalRegions = regionsY * regionsX

    actual = xp.empty(shape=(totalRegions, len(imgs)))

    for i in range(len(imgs)):
        for x in range(regionsX):
            for y in range(regionsY):
                x1 = int((x / regionsX) * (w - 1))
                x2 = int(((x + 1) / regionsX) * (w - 1))

                y1 = int((y / regionsY) * (h - 1))
                y2 = int(((y + 1) / regionsY) * (h - 1))
                
                actual[y * regionsX + x, i] = xp.mean(imgs[i, y1:y2, x1:x2])

    return actual

# def FlatField(blackImg, whiteImg):
    
#     firstImg = cam.images[0].rawData
#     lastImg = cam.images[-1].rawData

#     flatField = lastImg - firstImg

#     # Correct the images
#     fieldCorrectedImgs = [xp.clip((img - firstImg) / flatField, 0.0, 1.0) for img in imgs]  

@pytest.mark.skip(reason="Not ready")
def test_gamma():
    with ProcessingContext.UseGPU(True):
        xp = ProcessingContext().xp

        # Camera
        cam = camera.FileCamera(resolution=(1080, 1920), channels=1, refreshRate=30.0)

        # Parameters
        numImages = 51
        expected = np.linspace(0, 1.0, numImages, endpoint=True)
        # print(expected)

        for (w, h) in resolutions:
            proj = utils.FakeFPProjector(resolution=(h, w), channels=1, refreshRate=30.0, throwRatio=1.0, aspectRatio=1.0)

            expDir = expRoot / f"{w}x{h}"
            imgRepo = services.FileImageRepo(expDir, useExt='tif')

            cam.images = list(imgRepo.GetBy(f"gamma*", sorted=True))

            imgs = xp.asarray([camImg.rawData for camImg in cam.images])
            
            measured = RegionBasedIntensities(imgs, 3, 3).T

            print(xp.mean(measured[0]))
            print(xp.mean(measured[-1]))

# @pytest.mark.skip(reason="Not ready")
def test_calibration():
    with ProcessingContext.UseGPU(True):
        xp = ProcessingContext().xp

        # Camera
        cam = camera.FileCamera((1080, 1920), channels=1, refreshRate=30.0,
            character = characterisation.Characterisation(
                focalLengthGuess=(3.6, 3.6),
                sensorSizeGuess=(3.76, 2.115),
                opticalCentreGuess=(0.5, 0.5)
            )
        )

        # Parameters
        contrastMask = (0.0, 1.0/3.4)
        numImages = 12

        # Phase manipulation
        xShifter = shift.NStepPhaseShift([8, 8, 8])
        yShifter = shift.NStepPhaseShift([8, 8, 8])
        xUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0]) # Phase change perpendicular to baseline (x for this scenario)
        yUnwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0]) # Phase change follows baseline (y for this scenario)  

        # Characterisation board
        spacing = 3.0
        calibBoard = characterisation.CircleBoard(
            circleSpacing=(spacing, spacing), poiCount=(7, 21),
            inverted=True, staggered=True,
            areaHint=(500, 8000)
        )

        calibrator = calib.StereoCharacteriser(calibBoard, contrastMask=contrastMask)

        for (w, h) in resolutions:
            proj = utils.FakeFPProjector(resolution=(h, w), channels=1, refreshRate=30.0,
                throwRatio=1.0, aspectRatio=1.0,
                character = characterisation.Characterisation(
                    focalLengthGuess=(6.0/5.0, 6.0/5.0),
                    sensorSizeGuess=(1.0, 9.0/16.0),
                    opticalCentreGuess=(0.5, 0.5),
                ),
            )

            expDir = expRoot / f"{w}x{h}"
            imgRepo = services.FileImageRepo(expDir, useExt='tif')

            cam.images = list(imgRepo.GetBy(f"calibration*", sorted=True))

            phiXs = []
            phiYs = []
            camPOIs = []
            camImgs = []

            xFringeRot = 0.0

            # Gather all the images
            for i in range(numImages):

                phiX, camImg = calibrator.Measure(cam, proj, xShifter, xUnwrapper, xFringeRot)
                # Show(camImg, size=(1600, 900))
                # ShowPhasemap(phiX, size=(1600, 900))

                pois = calibBoard.FindPOIS(camImg)

                # Get y phasemap as successful
                phiY, _ = calibrator.Measure(cam, proj, yShifter, yUnwrapper, xFringeRot + (np.pi / 2.0), reverse=True)
                # ShowPhasemap(phiY, size=(1600, 900))

                if pois is None:
                    print(f"Could not identify POIs, skipping collection {i+1}")
                    Show(camImg, size=(1600, 900))
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
            services.FileCameraRepo(expDir).Add(cam, "camera")
            services.FileProjectorRepo(expDir).Add(proj, "projector")

            visionRepo = services.FileVisionConfigRepo(expDir)
            visionRepo.Add(cam.characterisation, "camera_vision")
            visionRepo.Add(proj.characterisation, "projector_vision")

@pytest.mark.skip(reason="Not ready")
def test_measurement():
    for (h, w) in resolutions[::-1]:
        testRoot = expRoot / f"{w}x{h}"

        with ProcessingContext.UseGPU(True):
            imageRepo = services.FileImageRepo(testRoot, useExt='tif')

            camRepo = services.FileCameraRepo(testRoot)
            projRepo = services.FileProjectorRepo(testRoot)
            visionRepo = services.FileVisionConfigRepo(testRoot)
            
            cam = camera.FileCamera(config=camRepo.Get("camera"), character=visionRepo.Get("camera_vision"))
            proj = utils.FakeFPProjector(config=projRepo.Get("projector"), visionConfig=visionRepo.Get("projector_vision"))

            reconstructor = recon.StereoReconstructor()
            shifter = shift.NStepPhaseShift([9, 9, 9], mask=0.1)
            unwrapper = unwrap.MultiFreqPhaseUnwrap([1.0, 8.0, 64.0])

            for obj in objects:
                cam.images = list(imageRepo.GetBy(obj, sorted=True))

                measurementCloud, dcImage, validPoints = reconstructor.Reconstruct(cam, proj, shifter, unwrapper, vertical=False)
                measurementCloud[:, 1] *= -1

                # Save and draw
                cloud.SaveArrayAsCloud(testRoot / f"{obj}.ply", measurementCloud)
                cloud.DrawCloud(cloud.ArrayToCloud(measurementCloud, colours=dcImage))