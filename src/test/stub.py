from opensfdi.devices import BaseProjector

from opensfdi import characterisation as ch

from pathlib import Path

DATA_ROOT = Path("D:\\results")

class StubProjector(BaseProjector):
    def __init__(self, resolution, channels, refresh_rate, throw_ratio, aspect_ratio, char:ch.ICharable=None):
        super().__init__(resolution, channels, refresh_rate, throw_ratio, aspect_ratio, char=char)

    def display(self, img):
        # Do nothing
        pass