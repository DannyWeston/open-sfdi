from sfdi.video import Projector, Camera

class Calibration:
    def __init__(self, width=1280, height=720):
        self.projector = Projector(width, height)
        self.camera = Camera()

    def run(self):
        pass