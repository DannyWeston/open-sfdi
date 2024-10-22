import logging

from time import sleep

class Photogrammetry:
    def __init__(self, cameras, delay):
        if len(cameras) < 2: raise Exception("You need at least 2 cameras to run an experiment") 
        
        self.logger = logging.getLogger(__name__)
        self.cameras = cameras
        self.delay = delay
        
    def run(self):
        if 0 < self.delay: sleep(self.delay)
        return [camera.capture() for camera in self.cameras]
