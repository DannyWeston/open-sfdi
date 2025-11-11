import pytest
import numpy as np

from opensfdi import services, image, calibration as calib
from opensfdi.devices import board, camera, projector
from opensfdi.phase import unwrap, shift

from . import utils

from pathlib import Path

def test_quicktest():
    expRoot = Path(f"D:\\results\\examples")
    imgRepo = services.FileImageRepo(expRoot, useExt='tif')

    cam = camera.FileCamera(camera.CameraConfig((1200, 1920), channels=1))
    proj = utils.FakeFPProjector(projector.ProjectorConfig(
        resolution=(1080, 1920), channels=1,
        throwRatio=1.4, aspectRatio=1.25)
    )
    
    cam.images = list(imgRepo.GetBy(f"calibration", sorted=True))

    calibBoard = board.CircleBoard(circleSpacing=(0.004, 0.004), 
        poiCount=(4, 13), inverted=True, staggered=True, poiMask=0.1)
    
    shifter = shift.NStepPhaseShift([9, 9, 9], mask=0.1)
    unwrapper = unwrap.MultiFreqPhaseUnwrap(
        numStripesVertical =    [1.0, 8.0, 64.0],
        numStripesHorizontal =  [1.0, 8.0, 64.0]
    )

    calibrator = calib.StereoCharacteriser(calibBoard)
    calibrator.debug = True
    calibrator.Characterise(cam, proj, shifter, unwrapper, poseCount=1)