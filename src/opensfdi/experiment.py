from abc import ABC, abstractmethod

import numpy as np
import logging

from time import sleep
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

from opensfdi.profilometry import ClassicPhaseHeight
from opensfdi.utils import maths

class Photogrammetry:
    def __init__(self, cameras, delay):
        if len(cameras) < 2: raise Exception("You need at least 2 cameras to run an experiment") 
        
        self.logger = logging.getLogger(__name__)
        self.cameras = cameras
        self.delay = delay
        
    def run(self):
        if 0 < self.delay: sleep(self.delay)
        return [camera.capture() for camera in self.cameras]

def fringe_projection(camera, projector, delay=0.0):
    if projector is None: 
        raise Exception("You need a projector to run fringe projection")
    
    if camera is None: 
        raise Exception("You need a camera to run fringe projection")
    
    projector.display()
    if 0 < delay: sleep(delay)
    return camera.capture()

def equal_phases(count):
    return [2.0 * np.pi * (i / count) for i in range(count)]

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

class Experiment(ABC):
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
        
        pass

class NStepFPExperiment(Experiment):
    def __init__(self, cameras, projector, steps):
        super().__init__()

        if projector is None: raise Exception("You need a projector to run fringe projection")
    
        if (cameras is None) or (len(cameras) == 0): raise Exception("You need a camera to run fringe projection")
        
        if steps < 3: raise Exception(f"You need at least 3 phases to run an N-step experiment ({steps} provided)")
        
        self.__projector = projector
        self.__cameras = cameras

        self._pre_cbs = []
        self._post_cbs = []

        self.__steps = steps

    def run(self):
        """ Run the experiment n times for both reference and measurement images """
        phases = equal_phases(self.__steps)

        # Run pre-reference image callbacks
        for cb in self._pre_cbs: cb()
        
        ref_imgs = []
        for i in range(self.__steps):
            self.__projector.phase = phases[i]
            ref_imgs.append([fringe_projection(c, self.__projector) for c in self.__cameras])
            
        # Run post-ref callbacks
        for cb in self._post_cbs: cb()
        
        imgs = []
        for i in range(self.__steps):
            self.__projector.phase = phases[i]
            imgs.append([fringe_projection(c, self.__projector) for c in self.__cameras])
        
        # TODO: Implement debug logger
        
        # Reorganise, not actually necessary
        ref_imgs = np.transpose(ref_imgs, (1, 0, 2, 3, 4))
        imgs = np.transpose(imgs, (1, 0, 2, 3, 4))
        
        return ref_imgs, imgs

    def classic_ph(self, ref_imgs, imgs, sf, cam_plane_dists, cam_proj_dists):
        cameras = imgs.shape[0]
        
        if len(cam_plane_dists) != cameras:
            raise Exception("You must provide a distance for all cameras to the ref plane")
        
        if len(cam_proj_dists) != cameras:
            raise Exception("You must provide a distance for all cameras to the projector")
        
        heightmaps = []
        
        for i in range(cameras):
            ph = ClassicPhaseHeight(sf, cam_plane_dists[i], cam_proj_dists[i])
            heightmap = ph.heightmap(ref_imgs[i], imgs[i], convert_grey=True, crop=None)
            
            heightmaps.append(heightmap)
        
        return heightmaps

    def add_pre_ref_callback(self, cb):
        self._pre_cbs.append(cb)
        
    def add_post_ref_callback(self, cb):
        self._post_cbs.append(cb)