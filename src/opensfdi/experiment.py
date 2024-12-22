import numpy as np

from typing import Callable

from opensfdi.phase import PhaseShift, PhaseUnwrap
from opensfdi.profilometry import BaseProf
from opensfdi.devices import Camera, FringeProjector

class Experiment:
    def __init__(self, name: str, profil: BaseProf, ph_shift: PhaseShift, ph_wrap: PhaseUnwrap):
        self.name = name

        self.streaming = False

        # Profilometry technique
        self.profil = profil

        # Phase-shift stuff
        self.ph_shift = ph_shift
        self.ph_unwrap = ph_wrap

        self.__on_height_measurement_cbs = []
        
        self.__post_phasemap_cbs : list[Callable] = []

    def calibrate(self, imgs, heights):
        """ Calibrate the experiment using a set of imgs at corresponding heights """
        # TODO: Implement debug logger

        # Single channel array of phasemaps
        ph, _, h, w, _ = imgs.shape
        phasemaps = np.ndarray(shape=((ph, h, w)), dtype=np.float64)

        for i, height in enumerate(heights):
            self.call_on_height_measurement(height)

            # Shift the captured images and unwrap them
            phasemap = self.ph_shift.shift(imgs[i])
            phasemap = self.ph_unwrap.unwrap(phasemap)
            phasemaps[i] = phasemap

            # Call post-phasemap cbs
            self.call_post_phasemap_cbs()

        # Run calibration
        self.profil.calibrate(phasemaps, heights)

    def get_imgs(self, camera: Camera, projector: FringeProjector):
        """ Run the experiment to gather the needed images """

        # TODO: Implement debug logger
        w, h = camera.resolution
        phase_count = self.ph_shift.phase_count
        
        imgs = np.ndarray(shape=((self.profil.phasemaps, phase_count, h, w, 3)), dtype=np.float64)
        for i in range(self.profil.phasemaps):
            imgs[i] = np.array([self.__fp_image(camera, projector, phase) for phase in self.ph_shift.get_phases()])

        return imgs
    
    def heightmap(self, imgs):
        """ Calculate a heightmap using passed imgs"""

        # Single channel array of phasemaps
        ph, _, h, w, _ = imgs.shape
        phasemaps = np.ndarray(shape=((ph, h, w)), dtype=np.float64)

        for i in range(self.profil.phasemaps):
            # Shift the captured images and unwrap them
            phasemap = self.ph_shift.shift(imgs[i])
            phasemap = self.ph_unwrap.unwrap(phasemap)
            phasemaps[i] = phasemap

            # Call post-phasemap cbs
            self.call_post_phasemap_cbs()

        # Obtain heightmaps
        return self.profil.heightmap(phasemaps)

    def stream(self):
        self.streaming = True
        
        while self.streaming:
            imgs = self.get_imgs()
            yield self.heightmap(imgs)
            
    def on_height_measurement(self, cb):
        """ Add a custom callback after taking a measurement at a certain height
         
            Before a measurement is taken at a certain height, stored callbacks are ran.
            The passed callback must accept the height as a parameter
        """
        self.__on_height_measurement_cbs.append(cb)


    # Callbacks

    def add_post_phasemap_cbs(self, cb: Callable):
        """ TODO: Add description """
        self.__post_phasemap_cbs.append(cb)

    def call_post_phasemap_cbs(self):
        for cb in self.__post_phasemap_cbs: cb()

    def call_on_height_measurement(self, height):
        for cb in self.__on_height_measurement_cbs: cb(height)

    # Private / dunder methods

    def __fp_image(self, camera: Camera, projector: FringeProjector, phase: float):
        projector.phase = phase
        projector.display()

        return camera.capture()

    def __str__(self):
        profil = type(self.profil)
        ph_shift = type(self.ph_shift)
        ph_unwrap = type(self.ph_unwrap)

        return f"{self.name} \n({profil}) \n({ph_shift}) \n({ph_unwrap})"