from opensfdi.devices import BaseProjector

from opensfdi import characterisation as ch

from pathlib import Path

DATA_ROOT = Path("D:\\results")

class StubProjector(BaseProjector):
    def __init__(self, resolution, channels, refresh_rate, throw_ratio, aspect_ratio, char:ch.ICharable=None):
        super().__init__(char=char)

        self._resolution = resolution
        self._channels = channels
        self._refresh_rate = refresh_rate
        self._throw_ratio = throw_ratio
        self._aspect_ratio = aspect_ratio

    @property
    def resolution(self):
        return self._resolution
    
    @property
    def channels(self):
        return self._channels
    
    @property
    def refresh_rate(self):
        return self._refresh_rate
    
    @property
    def throw_ratio(self):
        return self._throw_ratio
    
    @property
    def aspect_ratio(self):
        return self._aspect_ratio

    def display(self, img):
        # Do nothing
        pass