
import numpy as np
import cv2

import logging
import json
import os

from dataclasses import dataclass

from time import sleep
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

from sfdi.utils import maths

from sfdi.generation.fringes import Fringes
from sfdi.definitions import RESULTS_DIR

class LightCalc:
    def __init__(self, mu_a, mu_sp, refr_index, sf = [0.0, 0.2], std_dev = 3):
        self.mu_a = mu_a 
        self.mu_sp = mu_sp
        self.refr_index = refr_index
        self.sf = sf
        self.std_dev = std_dev
    
    def __calculate(self, mu_a, mu_sp, refr_index):
        # Effective Reflection Coefficient
        R_eff = 0.0636 * refr_index + 0.668 + 0.710 / refr_index - 1.44 / (refr_index ** 2)     
        
        A = (1 - R_eff) / (2 * (1 + R_eff)) # Proportionality Constant
        mu_tr = mu_sp + mu_a                # Transport Coefficient
        ap = mu_sp / mu_tr                  # Reduced albedo
        
        return R_eff, A, mu_tr, ap
    
    def calculate(self, imgs, ref_imgs):
        R_eff, A, mu_tr, ap = self.__calculate(self.mu_a, self.mu_sp, self.refr_index)
        
        ref_img_ac = maths.AC(ref_imgs)
        ref_img_dc = maths.DC(ref_imgs)
        
        # Apply some gaussian filtering if necessary
        if 0 < self.std_dev:
            ref_img_dc = gaussian_filter(ref_img_dc, self.std_dev)
            ref_img_ac = gaussian_filter(ref_img_ac, self.std_dev)

        # Get AC/DC Reflectance values using diffusion approximation
        r_ac, r_dc = maths.diffusion_approximation(A, ap, mu_tr, f[1])

        R_d_AC2 = (imgs_ac / ref_img_ac) * r_ac
        R_d_DC2 = (imgs_dc / ref_img_dc) * r_dc

        xi = []
        x, y = R_d_AC2.shape
        # Put the DC and AC diffuse reflectance values into an array
        for i in range(x):
            for j in range(y):
                freq = [R_d_DC2[i][j], R_d_AC2[i][j]]
                xi.append(freq)

        # Get an array of reflectance values and corresponding optical properties
        # We are setting the absorption coefficient range
        mu_a = np.arange(0, 0.5, 0.001)
        mu_sp = np.arange(0.1, 5, 0.01)

        # THE DIFFUSION APPROXIMATION
        # Getting the diffuse reflectance AC values corresponding to specific absorption and reduced scattering coefficients
        Reflectance_AC = []
        Reflectance_DC = []
        op_mua = []
        op_sp = []
        for i in range(len(mu_a)):
            for j in range(len(mu_sp)):
                # 1.43 is the refractive index of tissue
                R_eff, A, mu_tr, ap = self.__calculate(mu_a[i], mu_sp[j], 1.43)

                g = lambda mu_effp: (3 * A * ap) / (((mu_effp / mu_tr) + 1) * ((mu_effp / mu_tr) + 3 * A))

                ac = maths.mu_eff(mu_a[i], mu_tr, f[1])
                dc = maths.mu_eff(mu_a[i], mu_tr, f[0])

                Reflectance_AC.append(g(ac))
                Reflectance_DC.append(g(dc))

                op_mua.append(mu_a[i])
                op_sp.append(mu_sp[j])

        # putting the DC and AC diffuse reflectance values generated from the Diffusion Approximation into an array
        points = []
        for k in range(len(mu_a) * len(mu_sp)):
            freq = [Reflectance_DC[k], Reflectance_AC[k]]
            points.append(freq)

        points_array = np.array(points)
        
        #putting the optical properties into two seperate arrays
        op_mua_array = np.array(op_mua)
        op_sp_array = np.array(op_sp)

        #using scipy.interpolate.griddata to perform cubic interpolation of diffuse reflectance values to match
        #the generated diffuse reflectance values from image to calculated optical properties
        interp_method = 'cubic'
        coeff_abs = griddata(points_array, op_mua_array, xi, method=interp_method) #mua
        coeff_sct = griddata(points_array, op_sp_array, xi, method=interp_method) #musp

        abs_plot = np.reshape(coeff_abs, (R_d_AC2.shape[0], R_d_AC2.shape[1]))
        sct_plot = np.reshape(coeff_sct, (R_d_AC2.shape[0], R_d_AC2.shape[1]))

        absorption = np.nanmean(abs_plot)
        absorption_std = np.std(abs_plot)

        scattering = np.nanmean(sct_plot)
        scattering_std = np.std(sct_plot)


        self.logger.info(f'Absorption: {absorption}')
        self.logger.info(f'Deviation std: {absorption_std}')

        self.logger.info(f'Scattering: {scattering}')
        self.logger.info(f'Scattering std: {scattering_std}')

        return absorption, scattering, absorption_std, scattering_std

class ExpResult:
    def __init__(self, results=None, fringes=None, imgs=None, ref_imgs=None):
        self.results    = dict()
        self.fringes    = fringes
        self.imgs       = imgs
        self.ref_imgs   = ref_imgs
         
    def add_result(self, key, value):
        self.results[key] = value
        
    def remove_result(self, key):
        del self.results[key]

class Experiment:
    def __init__(self, cameras, projector, delay=0.0, ref_cbs=[], meas_cbs=[], debug=False):
        self.fp = FringeProjection(cameras, projector, delay, debug)
        self.logger = logging.getLogger("sfdi")
        
        self.ref_cbs = ref_cbs
        self.meas_cbs = meas_cbs

    def run(self, n=3):
        if n <= 0:
            self.logger.error("Number of measurements must be greater than 0")
            return None

        self.logger.info(f'Starting run')
        
        # Run the experiment n times for both reference and 
        ref_imgs = [self.fp.run() for _ in range(n)]

        # After finishing gathering reference images run callbacks
        for cb in self.ref_cbs: cb()

        imgs = [self.fp.run() for _ in range(n)]

        # After finishing gathering measurement images run callbacks
        for cb in self.meas_cbs: cb()
        
        self.logger.info(f'Run finished')
        
        return ExpResult(
                fringes=self.fp.projector,
                imgs=imgs,
                ref_imgs=ref_imgs
        )

        # if lightcalc:
        #     calc_time = perf_counter()
            
        #     mu_a, mu_s, mu_a_std, mu_s_std = self.calculate(imgs, ref_imgs)
            
        #     results["absorption"] = mu_a
        #     results["scattering"] = mu_s
        #     results["absorption_std_dev"] = mu_a_std
        #     results["scattering_std_dev"] = mu_s_std
    
        #     calc_time = perf_counter() - calc_time
        #     self.logger.info(f'Light calculations completed in {calc_time:.2f} seconds')

class FringeProjection:
    def __init__(self, cameras, projector, delay=0.0, debug=False):
        self.logger = logging.getLogger('sfdi')
        
        self.debug = debug

        self.cameras = cameras
        self.projector = projector
        self.delay = delay

    def __iter__(self):
        return self

    def __next__(self):
        self.__project_image()
        if 0.0 < self.delay: sleep(self.delay)
        return self.__collect_images()

    def run(self):
        self.__project_image()
        if 0 < self.delay: sleep(self.delay)
        return self.__collect_images()

    def stream(self):
        while True: yield self.run()

    def __project_image(self):
        if not self.projector: return
        
        if not self.projector.display(): raise StopIteration

    def __collect_images(self):        
        if self.debug:
        # TODO: Display camera preview so positioning can be accurate by user
            input("Press enter to take measurement")

        return [camera.capture() for camera in self.cameras]