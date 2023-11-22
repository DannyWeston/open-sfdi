
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import logging
import json
import os

from time import sleep

from time import perf_counter
from datetime import datetime

from sfdi.utils import maths
from sfdi.video import Camera

class Experiment:
    def __init__(self, args):
        self.proj_imgs = args["proj_imgs"]      # Projection images to use
        self.refr_index = args["refr_index"]    # Refractive index of material
        self.debug = args["debug"]              # Debug mode or not
        
        self.runs = args["runs"]                # How many runs to complete
        
        self.output_dir = os.path.join(args["output_dir"], datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Attempt to create results directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        elif 0 < len(os.listdir(self.output_dir)):
            raise FileExistsError(f'Experiment results directory {self.output_dir} is not empty!')

        self.mu_a = args["mu_a"]                # Reference absorption coefficient
        self.mu_sp = args["mu_sp"]              # Reference scattering coefficient

        self.camera_path = args["camera"]

        self.camera = Camera()

        self.logger = logging.getLogger()

    def start(self):
        start_time = perf_counter()

        successful = 0

        # Iterate through the runs, storing the results where necessary

        for i in range(1, self.runs + 1):
            self.logger.info(f'Starting run {i}')

            finish_time = perf_counter()

            results = self.__run()

            finish_time = perf_counter() - finish_time

            if results is None: continue

            successful += 1
            
            self.logger.info(f'Run {i} completed in {finish_time:.2f} seconds')

            self.save_results(results, self.output_dir, f'run{i}.json')

        start_time = perf_counter() - start_time

        self.logger.info(f'Experiment completed in {start_time:.2f} seconds ({successful}/{self.runs} successful)')

    def __run(self):
        # Collect 3 images from the camera
        # TODO: Add support for 3 =< images

        if self.debug: imgs, ref_imgs = self.test_images()
        else: imgs, ref_imgs = self.collect_images()

        f = [0, 0.2]

        # Calculate some constants

        R_eff = maths.ac_diffuse(self.refr_index)
        A = (1 - R_eff) / (2 * (1 + R_eff))
        mu_tr = self.mu_sp + self.mu_a
        ap = self.mu_sp / mu_tr

        std_dev = 3

        # Apply some gaussian filtering

        ref_imgs_ac = gaussian_filter(AC(ref_imgs), std_dev)
        ref_imgs_dc = gaussian_filter(DC(ref_imgs), std_dev)

        imgs_ac = gaussian_filter(AC(imgs), std_dev)
        imgs_dc = gaussian_filter(DC(imgs), std_dev)

        # Get AC/DC Reflectance values using diffusion approximation
        r_ac, r_dc = maths.diffusion_approximation(self.refr_index, self.mu_a, self.mu_sp, f[1])

        R_d_AC2 = (imgs_ac / ref_imgs_ac) * r_ac
        R_d_DC2 = (imgs_dc / ref_imgs_dc) * r_dc

        xi = []
        x, y = R_d_AC2.shape
        # Put the DC and AC diffuse reflectance values into an array
        for i in range(x):
            for j in range(y):
                freq = [R_d_DC2[i][j], R_d_AC2[i][j]]
                xi.append(freq)

        # Get an array of reflectance values and corresponding optical properties
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

        return {
            "absorption" : absorption,
            "absorption_std_dev" : absorption_std,
            "scattering" : scattering,
            "scattering_std_dev" : scattering_std,
        }

    def save_results(self, results, dir, name):
        with open(os.path.join(dir, name), 'w') as outfile:
            json.dump(results, outfile, indent=4)
            self.logger.info(f'Results saved as {name}')

    # Returns a list of n * 2 images (3 to use, 3 reference)
    def test_images(self):
        imgs = []
        ref_imgs = []

        img_paths = self.proj_imgs[:3]
        ref_img_paths = self.proj_imgs[3:]

        for path in img_paths:
            img = cv2.imread(path, 1).astype(np.double)
            img = img[:, :, 2] # Only keep red channel in images
            imgs.append(img)

        for path in ref_img_paths:
            ref_img = cv2.imread(path, 1).astype(np.double)
            ref_img = ref_img[:, :, 2] # Only keep red channel in images
            ref_imgs.append(ref_img)

        return imgs, ref_imgs

    def collect_images(self, fringe_patterns):
        imgs = []

        for pattern in fringe_patterns:
            cv2.namedWindow("main", cv2.WND_PROP_FULLSCREEN)          
            cv2.setWindowProperty("main", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("main", pattern)

            sleep(1) # Allow image warmup time

            img = self.camera.take_image() # Take a picture

            if not img:
                self.logger.error("Could not take an image using the camera")
                return None

            img = img[:, :, 2] # Apply some post processing (only keep red channel)
            
            imgs.append(img)

        return imgs