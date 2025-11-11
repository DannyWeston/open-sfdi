import pytest
import numpy as np

from opensfdi import services, image
from opensfdi.devices import board, camera

from pathlib import Path

# @pytest.mark.skip(reason="Not ready")
def test_calibration():
    expRoot = Path(f"C:\\Users\\Dan\\Desktop\\Digital Twin Demo\\Fake Images")
    imgRepo = services.FileImageRepo(expRoot, useExt='tif')

    cam = camera.FileCamera(camera.CameraConfig((3648, 5472), channels=3))
    
    boardImages = [x.rawData for x in imgRepo.GetBy(f"calibration", sorted=True)]

    calibBoard = board.CircleBoard(circleSpacing=(0.03, 0.03), poiCount=(4, 13), inverted=True, staggered=True, poiMask=0.1, areaHint=(2000, 10000))

    pixelCoords = np.asarray([calibBoard.FindPOIS(x) for x in boardImages])
    worldCoords = np.repeat(calibBoard.GetPOICoords()[np.newaxis, ...], len(boardImages), axis=0)

    cam.Characterise(worldCoords, pixelCoords)

    print(f"Camera reprojection error: {cam.characterisation.reprojErr}")

    services.FileCameraRepo(expRoot).Add(cam.config, "camera")
    services.FileVisionConfigRepo(expRoot).Add(cam.characterisation, "camera_vision")