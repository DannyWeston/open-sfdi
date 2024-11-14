import numpy as np

from abc import ABC, abstractmethod
from time import sleep

from opensfdi.phase import PhaseShift, PhaseUnwrap
from opensfdi.profilometry import BaseProf
from opensfdi.video import Camera, FringeProjector


class Experiment(ABC):
    @abstractmethod
    def __init__(self):
        self.streaming = False
    
        self.save_results = False
    
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def calibrate(self):
        pass

    def stream(self):
        self.streaming = True
        
        while self.streaming:
            yield self.run()

class FPExperiment(Experiment):
    def __init__(self, camera: Camera, projector: FringeProjector, ph_shift: PhaseShift, ph_wrap: PhaseUnwrap, calib: BaseProf, capture_delay = 0.0):
        self.__camera = camera
        self.__projector = projector

        self.__ph_shift = ph_shift
        self.__ph_unwrap = ph_wrap

        self.__calib = calib

        self.__capture_delay = capture_delay

        self.__on_height_measurement_cbs = []

    @property
    def calibration(self):
        return self.__calib

    def __fp_image(self, phase):
        # Get all the required images
        self.__projector.phase = phase
        self.__projector.display()

        return self.__camera.capture()

    def calibrate(self, heights):
        """ Calibrate the experiment using a set of provided heights """
        # TODO: Implement debug logger

        phasemaps = []

        # TODO: Implement debug logger
        for height in heights:
            self.call_on_height_measurement(height)

            # Gather a sequence of imagess
            imgs = np.array([self.__fp_image(phase) for phase in self.__ph_shift.get_phases()])

            # Shift the captured images and unwrap them
            phasemap = self.__ph_shift.shift(imgs)
            phasemap = self.__ph_unwrap.unwrap(phasemap)
            phasemaps.append(phasemap)

            # Call post-phasemap cbs
            self.__calib.call_post_cbs()

            # Sleep for delay if needed
            if 0 < self.__capture_delay: sleep(self.__capture_delay)

        # Run calibration
        self.__calib.calibrate(np.array(phasemaps), heights)
    
    def run(self):
        """ Run the experiment to gather the needed images """

        # TODO: Implement debug logger
        phasemaps = []
        for _ in range(self.__calib.meas_ph_needed):
            # Gather a sequence of imagess
            imgs = np.array([self.__fp_image(phase) for phase in self.__ph_shift.get_phases()])
            
            # # Shift the captured images and unwrap them
            phasemap = self.__ph_shift.shift(imgs)
            phasemap = self.__ph_unwrap.unwrap(phasemap)
            phasemaps.append(phasemap)

            # Call post-phasemap cbs
            self.__calib.call_post_cbs()

        if len(phasemaps) != self.__calib.meas_ph_needed:
            raise Exception("Number of phasemaps generated does not match what the calibration requires")

        # Obtain heightmaps
        return self.__calib.heightmap(np.array(phasemaps))

    def on_height_measurement(self, cb):
        """ Add a custom callback after taking a measurement at a certain height
         
            Before a measurement is taken at a certain height, stored callbacks are ran.
            The passed callback must accept the height as a parameter
        """
        self.__on_height_measurement_cbs.append(cb)

    def call_on_height_measurement(self, height):
        for cb in self.__on_height_measurement_cbs: cb(height)