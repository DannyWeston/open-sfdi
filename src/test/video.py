from opensfdi.devices import Projector, ProjectorRegistry

@ProjectorRegistry.register
class FakeProjector(Projector):
    def __init__(self, resolution=(720, 1280), channels=1):
        super().__init__(resolution=resolution, channels=channels)

    def display(self):
        # Do nothing
        pass