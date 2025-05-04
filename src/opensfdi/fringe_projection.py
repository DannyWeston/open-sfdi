import cv2
import numpy as np

from time import perf_counter
from typing import Callable

from . import phase, profilometry, image

class FringeProjection:
    def __init__(self, shifter: phase.PhaseShift, unwrapper: phase.PhaseUnwrap):
        # Phase-shift stuff
        self.ph_shift = shifter
        self.ph_unwrap = unwrapper

        # Metrics
        self.calibration_time = 0.0

        # Callbacks
        self.__on_height_measurement_cbs = []
        self.__post_phasemap_cbs : list[Callable] = []

    def calculate_mask(self, img_data, min_thresh=3, max_thresh=255):
        img_data = image.to_grey(image.to_int8(img_data))

        _, mask = cv2.threshold(img_data, min_thresh, max_thresh, cv2.THRESH_BINARY)
        img_data = cv2.bitwise_and(img_data, img_data, mask=mask)

        return img_data, mask

    # Stereo method

    # def measure_stereo(self, reconstructor: profilometry.IStereoReconstructor) -> np.ndarray:
    #     # Setup devices
    #     camera = reconstructor.camera
    #     projector = reconstructor.projector

    #     projector.rotation = reconstructor.get_rotation() # Set projector fringe rotation

    #     dc_img = None

    #     shifted = []

    #     for i, imgs in enumerate(self.__capture_phase_set(camera, projector)):
    #         if i == 0: dc_img = image.dc_imgs(imgs)

    #         # TODO: Investigate using multi-channel fringe projection
    #         grey_imgs = np.array([image.to_grey(img) for img in imgs])

    #         # Phase shift the images
    #         shifted.append(self.ph_shift.shift(grey_imgs))

    #     # Unwrap the shifted images
    #     phasemap = self.ph_unwrap.unwrap(shifted)

    #     # Obtain reconstructed cloud
    #     cloud = reconstructor.reconstruct(phasemap, projector.frequency)

    #     profilometry.show_pointcloud(cloud, colours=cv2.cvtColor(dc_img, cv2.COLOR_BGR2RGB))

    #     return cloud

    def calibrate_phaseheight(self, calibrator: profilometry.IPhaseHeightCalibrator):
        return None

    def measure_phaseheight(self):
        """ Calculate a heightmap using passed imgs"""

        # Single channel array of phasemaps
        ph, N, h, w = imgs.shape

        if N != self.ph_shift.phase_count: raise Exception("Incorrect number of images for phase shifting")
        if ph != self.profil.phasemaps: raise Exception("Incorrect number of phasemaps for profilometry")
    
        phasemaps = np.ndarray(shape=((ph, h, w)), dtype=np.float32)

        for i in range(self.profil.phasemaps): # Shift the captured images and unwrap them
            wrapped_phasemap = self.ph_shift.shift(imgs[i])
            phasemaps[i] = self.ph_unwrap.unwrap(wrapped_phasemap)

            # Call post-phasemap cbs
            self.call_post_phasemap_cbs()

        # Obtain heightmaps
        return self.profil.heightmap(phasemaps)
    #     """ Calibrate the experiment using a set of imgs at corresponding heights """
    #     start_time = perf_counter()

    #     while len(imgs) != 0:
    #         for f in range(unwrapper.phase_count):
    #             phi_w = []

    #             # Take correct number of images for phase shifting
    #             n_imgs = imgs[:shifter.phase_count]
    #             imgs = imgs[shifter.phase_count:]

    #             raw_imgs = np.array([img.data for img in n_imgs])
    #             grey_imgs = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in raw_imgs])
                
    #             if unwrapper.phase_count == 1:
    #                 shifted = shifter.shift(grey_imgs)
    #                 dc_imgs = image.dc_imgs(grey_imgs)
    #             else:
    #                 l = shifter.phase_count
    #                 dc_imgs = [image.dc_imgs(grey_imgs[i * l::l]) for i in range(unwrapper.phase_count)]
    #                 shifted = [shifter.shift(grey_imgs[i * l::l]) for i in range(unwrapper.phase_count)]

                
                
    #             img_size = dc_imgs[0].shape
    #             corners = self.__find_corners(dc_imgs[0], cb_corners, self.cb_size, self.window_size)

    #             if corners is None:
    #                 print(f"Could not find checkerboard corners")
    #                 continue

                

    #         w_xyz.append(cb_corners)
    #         c_xyz.append(corners)


    #         unwrapped = unwrapper.unwrap(shifted)
    #         phasemaps.append(unwrapped)

    #     # Images should have a single channel only
    #     # Single channel array of phasemaps
    #     # TODO: Support multi-channel images
    #     z, phases, h, w = imgs.shape

    #     if phases != self.ph_shift.phase_count:
    #         raise Exception(f"""There aren't sufficient images at each height for phase
    #                             shifting ({phases} given,{self.ph_shift.phase_count} required))""")

    #     if z != len(heights):
    #         raise Exception("Number of phasemaps does not match number of heights passed")
        
    #     z_phasemaps = np.ndarray(shape=((z, h, w)), dtype=np.float32)

    #     for i, height in enumerate(heights):
    #         self.call_on_height_measurement(height)

    #         # Shift the captured images and unwrap them
    #         wrapped_phasemap = self.ph_shift.shift(imgs[i])
    #         z_phasemaps[i] = self.ph_unwrap.unwrap(wrapped_phasemap)

    #         # Call post-phasemap cbs
    #         self.call_post_phasemap_cbs()

    #     # Run calibration
    #     self.profil.calibrate(z_phasemaps, heights)

    #     # Show calibration time
    #     print(f"Calibration took {(perf_counter() - start_time):.2f}s")

    # def calibrate_imgs(self, camera: Camera, projector: Projector, heights: np.ndarray):
    #     """ Run the experiment to gather the needed images """
    #     start_time = perf_counter()

    #     w, h = camera.resolution
    #     phases = self.ph_shift.get_phases()
        
    #     # Preallocate images memory
    #     imgs = np.ndarray(shape=((len(heights), len(phases), h, w)), dtype=np.uint8)
        
    #     for i, height in enumerate(heights):
    #         self.call_on_height_measurement(height)

    #         for j, phase in enumerate(phases):
    #             # Get a unwrapped phasemap at each height
    #             img = self.get_img(camera, projector, phase)

    #             # If using 3-channel images, convert to use only red channel for now
    #             # TODO: Support 3-channel fringe projection (or more!)
    #             if camera.channels != 1:
    #                 img = img[..., 0].reshape(h, w)

    #             imgs[i, j] = img


    #     print(f"{len(heights) * len(phases)} images captured in {(perf_counter() - start_time):.2f}s")

    #     return imgs


    # Callbacks

    def on_height_measurement(self, cb):
        """ Add a custom callback after taking a measurement at a certain height
         
            Before a measurement is taken at a certain height, stored callbacks are ran.
            The passed callback must accept the height as a parameter
        """
        self.__on_height_measurement_cbs.append(cb)

    def call_on_height_measurement(self, height):
        for cb in self.__on_height_measurement_cbs: cb(height)

    def add_post_phasemap_cbs(self, cb: Callable):
        """ TODO: Add description """
        self.__post_phasemap_cbs.append(cb)

    def call_post_phasemap_cbs(self):
        for cb in self.__post_phasemap_cbs: cb()

    def __str__(self):
        ph_shift = type(self.ph_shift)
        ph_unwrap = type(self.ph_unwrap)

        return f"""<Fringe Projection> \nPhase-shifting:({ph_shift}) \nPhase-unwrapping({ph_unwrap})"""