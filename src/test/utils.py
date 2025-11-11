from opensfdi.devices import projector, vision

from pathlib import Path

DATA_ROOT = Path("D:\\results")

class FakeFPProjector(projector.FringeProjector):
    def __init__(self, resolution, channels, refreshRate, throwRatio, aspectRatio, character: vision.Characterisation=None, 
                stripeRotation=0.0, phase=0.0, numStripes=0.0):
        
        super().__init__(resolution, channels, refreshRate, throwRatio, aspectRatio, 
                stripeRotation, phase, numStripes,
                character=character)

    def Display(self):
        # Do nothing
        pass