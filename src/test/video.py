from opensfdi.devices import Camera, Projector
from opensfdi import image

class FakeCamera(Camera):
    def __init__(self, imgs: list[image.FileImage], resolution=(720, 1280), channels=3):
        super().__init__(resolution, channels)

        self.__imgs = imgs

    @property
    def imgs(self):
        return self.__imgs
    
    @imgs.setter
    def imgs(self, value):
        self.__imgs = value

    def capture(self) -> image.Image:
        try:
            return self.__imgs.pop(0)
        except IndexError:
            return None

class FakeProjector(Projector):
    def __init__(self, resolution=(720, 1280)):
        self.__resolution = resolution

        self.__frequency = 0.0
        self.__rotation = True
        self.__phase = 0.0

    @property
    def frequency(self) -> int:
        return self.__frequency
    
    @frequency.setter
    def frequency(self, value):
        self.__frequency = value

    @property
    def resolution(self) -> tuple[int, int]:
        return self.__resolution
        
    @property
    def rotation(self) -> bool:
        return self.__rotation
    
    @rotation.setter
    def rotation(self, value: bool) -> None:
        self.__rotation = value
    
    @property
    def phase(self) -> float:
        return self.__phase
    
    @phase.setter
    def phase(self, value) -> float:
        self.__phase = value

    def display(self):
        pass