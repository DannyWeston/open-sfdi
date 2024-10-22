import numpy as np

from abc import ABC, abstractmethod
from time import sleep

from opensfdi.phase_shifting import PhaseShift
from opensfdi.phase_unwrapping import PhaseUnwrap
from opensfdi.profilometry.phase import PhaseHeight
from opensfdi.video import Camera, FringeProjector

class Experiment(ABC):
    @abstractmethod
    def __init__(self):
        self.streaming = False
    
        self.save_results = False
    
    @abstractmethod
    def run(self):
        pass

    def stream(self):
        self.streaming = True
        
        while self.streaming:
            yield self.run()

class FPExperiment(Experiment):
    def __init__(self, camera: Camera, projector: FringeProjector, ph_shift: PhaseShift, ph_wrap: PhaseUnwrap, calib: PhaseHeight, capture_delay: 0.0):
        self.__camera = camera
        self.__projector = projector

        self.__ph_shift = ph_shift
        self.__ph_unwrap = ph_wrap

        self.__calib = calib

        self.__capture_delay = capture_delay

    def __capture_sequence(self):
        # Get all the required images
        imgs = []
        phases = self.__ph_shift.get_phases()

        for i in range(self.__ph_shift.required_imgs):
            # Display the phase on the fringe projector
            self.__projector.phase = phases[i]
            self.__projector.display()
            
            # Sleep for delay if needed
            if 0 < self.__capture_delay: sleep(self.__capture_delay)

            # Capture the image with the camera
            imgs.append(self.__camera.capture())

        return imgs

    def run(self):
        """ Run the experiment to gather the needed images """

        # TODO: Implement debug logger

        imgs = []
        for _ in self.__calib.phasemaps_needed:
            # Gather a sequence of images
            imgs.append(self.__capture_sequence())

            # Run any callbacks that need to be processed
            self.__calib.call_post_cbs()

        # Shift the captured images and unwrap them
        phasemaps = [self.__ph_shift.shift(xs) for xs in imgs]
        phasemaps = np.array([self.__ph_unwrap.unwrap(pm) for pm in phasemaps])

        if len(phasemaps) != self.__calib.phasemaps_needed:
            raise Exception("Phasemaps generated does not match what the calibration requires")

        # Gather profilometry information
        return self.__calib.heightmap(phasemaps)