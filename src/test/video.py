import numpy as np

from opensfdi.devices import Camera, FringeProjector

class FakeCamera(Camera):
    def __init__(self, imgs:np.ndarray, circular=False):
        self.__imgs = imgs

        self.__index = 0

        self.__circular = circular

    @property
    def resolution(self) -> tuple[int, int]:
        return (0, 0)

    @property
    def distortion(self) -> object:
        return None
    
    def capture(self):
        l = self.__imgs.shape[0]

        if 0 <= self.__index < l:
            img = self.__imgs[self.__index]
            self.__index += 1
            return img
        
        elif self.__circular:
            self.__index = 0 

        raise RuntimeError("There are no images to use")


class FakeFringeProjector(FringeProjector):
    def __init__(self):
        pass

    @property
    def frequency(self) -> int:
        return 0

    @property
    def resolution(self) -> tuple[int, int]:
        return (0, 0)
        
    @property
    def rotation(self) -> float:
        return 0.0
    
    @property
    def phase(self) -> float:
        return 0.0
    
    @phase.setter
    def phase(self, value) -> float:
        pass

    def display(self):
        """ Does nothing """
        pass