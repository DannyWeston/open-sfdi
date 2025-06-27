from opensfdi.devices import projector

class FakeFPProjector(projector.FringeProjector):
    def __init__(self, config: projector.ProjectorConfig, stripeRotation=0.0, phase=0.0, numStripes=0.0):
        super().__init__(config, stripeRotation, phase, numStripes)

    def Display(self):
        # Do nothing
        pass