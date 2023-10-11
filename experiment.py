
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy import ndimage, misc
import pandas as pd
import logging

from camera import Camera

f = [0, 0.2]

# Demodulation (array input)
def AC(var: list):
    return ((2) ** (1 / 2) / 3) * (((var[0] - var[1]) ** 2 + (var[1] - var[2]) ** 2 + (var[2] - var[0]) ** 2) ** (1 / 2))

def DC(var: list):
    return (1 / 3) * (var[0] + var[1] + var[2])

####################REFERENCE PHANTOM OPTICAL PROPERTIES###########################
mu_a = 0.018
mu_sp = 0.77
n = 1.43 #refractive index of tissue 
###################################################################################


#AC diffuse reflectance from Diffusion Approximation
R_eff = 0.0636 * n + 0.668 + 0.710 / n - 1.44 / (n ** 2)
A = (1 - R_eff) / (2 * (1 + R_eff))
mu_tr = mu_a + mu_sp
ap = mu_sp / mu_tr

def mu_eff(mu_a, mu_tr, ac=False):
    return mu_tr * (3 * (mu_a / mu_tr) + ((2 * np.pi * f[0] if ac else f[1]) ** 2) / mu_tr ** 2) ** 0.5


def diffusion_approximation():
    #AC diffuse reflectance from Diffusion Approximation
    R_eff = 0.0636 * n + 0.668 + 0.710 / n - 1.44 / (n ** 2)
    A = (1 - R_eff) / (2 * (1 + R_eff))
    mu_tr = mu_a + mu_sp
    ap = mu_sp / mu_tr

    r_ac = (3 * A * ap)/(((2 * np.pi * f[1]) / mu_tr) ** 2 + ((2 * np.pi * f[1]) / mu_tr) * (1 + 3 * A) + 3 * A)
    r_dc = (3 * A * ap) / (3 * (1 - ap) + (1 + 3 * A) * np.sqrt(3 * (1 - ap)) + 3 * A)

    return r_ac, r_dc

class Experiment:
    def __init__(self, proj_imgs, camera, img_func = None, debug=False):
        self.proj_imgs = proj_imgs
        self.img_func = img_func

        self.logger = logging.getLogger()

        self.camera = Camera(camera)

        self.camera.show_raw_feed()

        self.debug = debug

    def run(self, run_id):
        self.logger.info(f'Starting run {run_id + 1}')

        imgs = []
        ref_imgs = []

        # Try and load the projector images, 3 phrases
        for proj_img in self.proj_imgs:
            img = self.__load_img(proj_img)

            if self.img_func: img = self.img_func(img) # Apply some filtering to image if valid
            
            imgs.append(img)
            ref_imgs.append(img)

        std_dev = 3

        ref_imgs_ac = gaussian_filter(AC(ref_imgs), std_dev)
        ref_imgs_dc = gaussian_filter(DC(ref_imgs), std_dev)

        imgs_ac = gaussian_filter(AC(imgs), std_dev)
        imgs_dc = gaussian_filter(DC(imgs), std_dev)

        # Get AC/DC Reflectance values using diffusion approximation
        r_ac, r_dc = diffusion_approximation()

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
                mu_effp_AC = mu_tr * (3 * (mu_a[i] / mu_tr) + ((2 * np.pi * f[1]) ** 2) / mu_tr ** 2) ** 0.5
                R_d1 = (3 * A * ap) / (((mu_effp_AC / mu_tr) + 1) * ((mu_effp_AC / mu_tr) + 3 * A))
                Reflectance_AC.append(R_d1)
                
                mu_effp_DC = mu_eff()
                R_d2 = (3 * A * ap) / (((mu_effp_DC / mu_tr) + 1) * ((mu_effp_DC / mu_tr) + 3 * A))
                Reflectance_DC.append(R_d2)
                        
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

        print("Absorption : ", np.nanmean(abs_plot))
        print("Absorption std: ", np.std(abs_plot))
        print("Scattering: ", np.nanmean(sct_plot))
        print("Scattering std: ", np.std(sct_plot))
    
        self.logger.info(f'Completed run {run_id + 1}')

    def __load_img(self, path):
        img = cv2.imread(path, 1)

        if self.debug: self.__display_img(img)

        return img.astype(np.double)
    
    def __display_img(self, img):
        cv2.namedWindow("main", cv2.WND_PROP_FULLSCREEN)          
        cv2.setWindowProperty("main", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("main", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()