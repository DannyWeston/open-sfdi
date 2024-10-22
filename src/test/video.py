from opensfdi.video import Camera, FringeProjector

class FakeCamera(Camera):
    def __init__(self, imgs=[], consume=False):
        self.__imgs = imgs

        self.__consume = consume

    @property
    def resolution(self) -> tuple[int, int]:
        return (0, 0)

    @property
    def distortion(self) -> object:
        return None
    
    def capture(self):
        if len(self.__imgs) == 0: raise RuntimeError("There are no images left to use")

        next_img = self.__imgs.pop(0)

        # Readd the image to the queue if consuming is disabled
        if not self.__consume: self.__imgs.append(next_img)

        return next_img

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

    def display(self):
        """ Does nothing """
        pass