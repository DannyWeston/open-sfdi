
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy import ndimage, misc
import pandas as pd
import logging

from time import perf_counter

from maths import *
from camera import Camera

f = [0, 0.2]

class Experiment:
    def __init__(self, args, img_func = None):
        self.proj_imgs = args["proj_imgs"]      # Projection images to use
        self.refr_index = args["refr_index"]    # Refractive index of material
        self.debug = args["debug"]              # Debug mode

        self.mu_a = args["mu_a"]                # Reference absorption coefficient
        self.mu_sp = args["mu_sp"]              # Reference scattering coefficient

        self.camera = Camera(args["camera"])

        self.img_func = img_func

        if self.debug: self.start_time = None

        self.logger = logging.getLogger()

    def run(self, run_id):
        self.logger.info(f'Starting run {run_id + 1}')

        if self.debug:
            self.start_time = perf_counter()

        # Calculate some constants

        R_eff = ac_diffuse(self.refr_index)
        A = (1 - R_eff) / (2 * (1 + R_eff))
        mu_tr = self.mu_sp + self.mu_a
        ap = self.mu_sp / mu_tr

        imgs, ref_imgs = self.load_images()

        std_dev = 3

        ref_imgs_ac = gaussian_filter(AC(ref_imgs), std_dev)
        ref_imgs_dc = gaussian_filter(DC(ref_imgs), std_dev)

        imgs_ac = gaussian_filter(AC(imgs), std_dev)
        imgs_dc = gaussian_filter(DC(imgs), std_dev)

        # Get AC/DC Reflectance values using diffusion approximation
        r_ac, r_dc = diffusion_approximation(self.refr_index, self.mu_a, self.mu_sp, f[1])

        R_d_AC2 = (imgs_ac / ref_imgs_ac) * r_ac
        R_d_DC2 = (imgs_dc / ref_imgs_dc) * r_dc

        xi = []
        x, y = R_d_AC2.shape
        #puts the DC and AC diffuse reflectance values into an array
        for i in range(x):
            for j in range(y):
                freq = [R_d_DC2[i][j], R_d_AC2[i][j]]
                xi.append(freq)

        #Getting array of reflectance values and corresponding optical properties
        mu_a = np.arange(0, 0.5, 0.001) # We are setting the absorption coefficient range
        mu_sp = np.arange(0.1, 5, 0.01)

        n = 1.43 #refractive index of tissue

        # THE DIFFUSION APPROXIMATION
        # Getting the diffuse reflectance AC values corresponding to specific absorption and reduced scattering coefficients
        Reflectance_AC = []
        Reflectance_DC = []
        op_mua = []
        op_sp = []
        for i in range(len(mu_a)):
            for j in range(len(mu_sp)):
                R_eff = 0.0636 * n + 0.668 + 0.710 / n - 1.44 / (n ** 2)
                A = (1 - R_eff) / (2 * (1 + R_eff))
                mu_tr = mu_a[i] + mu_sp[j]
                ap = mu_sp[j] / mu_tr

                g = lambda mu_effp: (3 * A * ap) / (((mu_effp / mu_tr) + 1) * ((mu_effp / mu_tr) + 3 * A)) 

                ac = mu_eff(mu_a[i], mu_tr, f[1])
                dc = mu_eff(mu_a[i], mu_tr, f[0])

                Reflectance_AC.append(g(ac))
                Reflectance_DC.append(g(dc))

                op_mua.append(mu_a[i])
                op_sp.append(mu_sp[j])

        #putting the DC and AC diffuse reflectance values generated from the Diffusion Approximation into an array    
        points = []
        for k in range(len(mu_a) * len(mu_sp)): 
            freq = [Reflectance_DC[k], Reflectance_AC[k]]
            points.append(freq)

        points_array = np.array(points)
        #putting the optical properties into two seperate arrays
        op_mua_array = np.array(op_mua) #values1
        op_sp_array = np.array(op_sp) #values2

        #using scipy.interpolate.griddata to perform cubic interpolation of diffuse reflectance values to match 
        #the generated diffuse reflectance values from image to calculated optical properties
        interp_method = 'cubic'
        coeff_abs = griddata(points_array, op_mua_array, xi, method=interp_method) #mua
        coeff_sct = griddata(points_array, op_sp_array, xi, method=interp_method) #musp

        abs_plot = np.reshape(coeff_abs, (R_d_AC2.shape[0], R_d_AC2.shape[1]))
        sct_plot = np.reshape(coeff_sct, (R_d_AC2.shape[0], R_d_AC2.shape[1]))

        if self.debug:
            self.logger.info(f'Run took ')

        self.logger.info(f'Absorption: {np.nanmean(abs_plot)}')
        self.logger.info(f'Deviation std: {np.std(abs_plot)}')

        self.logger.info(f'Scattering: {np.nanmean(sct_plot)}')
        self.logger.info(f'Scattering std: {np.std(sct_plot)}')

        finish_time = perf_counter() - self.start_time

        self.logger.info(f'Completed run {run_id + 1} in {finish_time:.2f} seconds')

    def load_images(self):
        imgs = []
        ref_imgs = []

        img_paths = self.proj_imgs[:3]
        ref_img_paths = self.proj_imgs[3:]

        for path in img_paths:
            img = self.__load_img(path)
            if self.img_func: img = self.img_func(img) # Apply some filtering to image if valid
            imgs.append(img)

        for path in ref_img_paths:
            ref_img = self.__load_img(path)
            if self.img_func: ref_img = self.img_func(ref_img) # Apply some filtering to image if valid
            ref_imgs.append(ref_img)

        return imgs, ref_imgs

    def __load_img(self, path):
        img = cv2.imread(path, 1)

        return img.astype(np.double)
    
    def __display_img(self, img):
        cv2.namedWindow("main", cv2.WND_PROP_FULLSCREEN)          
        cv2.setWindowProperty("main", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("main", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()