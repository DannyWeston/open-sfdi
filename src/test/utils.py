from opensfdi.devices import projector, vision

from pathlib import Path

DATA_ROOT = Path("D:\\results")

class FakeFPProjector(projector.FringeProjector):
    def __init__(self, config: projector.ProjectorConfig, visionConfig:vision.VisionConfig=None):
        super().__init__(config, visionConfig=visionConfig)

    def Display(self):
        # Do nothing
        pass