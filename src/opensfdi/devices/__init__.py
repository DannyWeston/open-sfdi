
from . import camera, projector

from .. import image

def FringeProject(camera: camera.Camera, projector: projector.FringeProjector, numStripes: float, phase: float) -> image.Image:
    projector.numStripes = numStripes
    projector.phase = phase
    projector.Display()

    return camera.Capture()