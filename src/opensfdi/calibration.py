import cv2
import logging
import numpy as np

import matplotlib.pyplot as plt

from opensfdi import DEBUG

from abc import ABC, abstractmethod

def apply_correction(img, coeffs, x1=0.0, x2=1.0):
    poly_func = np.poly1d(coeffs)

    corrected_img = poly_func(img)

    corrected_img[corrected_img < x1] = x1 # Cutoff values less than x1
    corrected_img[corrected_img > x2] = x2 # Cutoff values greater than x2

    return corrected_img

class Calibration(ABC):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def calibrate(self):
        raise NotImplementedError

class GammaCalibration(Calibration):
    def __init__(self, camera, projector, delta, crop_size=0.25, order=5, intensity_count=32):
        super().__init__()
        
        self.camera = camera
        self.projector = projector
        
        self._delta = delta
        self._crop_size = crop_size
        self._order = order
        self._intensity_count = intensity_count
        
        self.coeffs = None
        self.visible = None

    def calibrate(self):
        intensities = np.linspace(0.0, 1.0, self._intensity_count, dtype=np.float32)
        w, h = self.camera.resolution
        self.projector.imgs = np.array([(np.ones((h, w, 3), dtype=np.float32) * i) for i in intensities]) # 3 channels for rgb
        
        # Need to get some images
        captured_imgs = np.empty((self._intensity_count,), dtype=np.ndarray)
        
        # Capture all of the images
        for i in range(self._intensity_count):
            self.projector.display()
            captured_imgs[i] = self.camera.capture()
        
        cap_height, cap_width, _ = captured_imgs[0].shape

        # Calculate region of interest values
        roi = int(cap_width * self._crop_size)
        mid_height = int(cap_height / 2)
        mid_width = int(cap_width / 2)
        rows = [x + mid_height for x in range(-roi, roi)]
        cols = [x + mid_width for x in range(-roi, roi)]

        # Calculate average pixel value for each image
        averages = [np.mean(x[rows, cols]) for x in captured_imgs]

        # Find sfirst observable change of values for averages (left and right sides) i.e >= delta
        s, f = self._detectable_indices(averages, self._delta)

        vis_averages = averages[s:f+1]
        vis_intensities = intensities[s:f+1]

        self.coeffs = np.polyfit(vis_averages, vis_intensities, self._order)
        self.visible = intensities[s:f+1]

        if DEBUG: # Plot results
            plt.plot(vis_averages, vis_intensities, 'o')
            trendpoly = np.poly1d(self.coeffs)
            plt.title('Gamma Calibration Curve Results')
            
            plt.xlabel("Measured")
            plt.ylabel("Actual")
            
            plt.plot(vis_averages, trendpoly(vis_averages))
            plt.show()

        return self.coeffs, self.visible

    def serialize(self):
        return {
                "coeffs"                : self.coeffs.tolist(),
                "visible_intensities"   : self.visible.tolist()
            }
        
    def deserialize(self):
        return None

    def _detectable_indices(self, values, delta):
        start = finish = None

        for i in range(1, len(values) - 1):
            x1 = values[i - 1]
            x2 = values[i]

            y1 = values[len(values) - i - 1]
            y2 = values[len(values) - i]

            if not start and abs(x1 - x2) >= delta:
                start = i

            if not finish and abs(y1 - y2) >= delta:
                finish = len(values) - i - 1

        return start, finish

class CameraCalibration(Calibration):
    def __init__(self, camera, img_count=10, cb_size=(8, 6)):
        super().__init__()
        
        self.camera = camera
        
        self._cb_size = cb_size
        
        if img_count < 10:
            raise Exception("10 or more images are required to calibrate cameras")

        self._img_count = img_count
        
        self.cam_mat = self.dist_mat = self.optimal_mat = None

    def on_checkboard_change(self, i):
        '''
            (i) = Image number just completed (doesn't take into account multiple cameras)
        '''
        pass

    def calibrate(self):
        self.logger.info(f"Using {self._img_count} checkerboard images to calibrate {self.camera.name}")
        
        imgs = np.empty((self._img_count), dtype=np.ndarray)
        
        for i in range(len(imgs)):
            img = self.camera.capture()
            imgs[i] = img

            self.on_checkerboard_change(i)
        
        CHECKERBOARD = (self._cb_size[0] - 1, self._cb_size[1] - 1)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        threedpoints = []
        twodpoints = []

        objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        for i, image in enumerate(imgs):
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(grey_img, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                threedpoints.append(objectp3d)

                corners2 = cv2.cornerSubPix(grey_img, corners, (11, 11), (-1, -1), criteria)

                twodpoints.append(corners2)
                
                #image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
            else:
                self.logger.warning(f'{self.camera.name} failed to find checkerboard on image {i}')

        h, w = imgs[0].shape[:2]

        ret, self.cam_mat, self.dist_mat, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grey_img.shape[::-1], None, None)

        if not ret: raise

        self.optimal_mat, roi = cv2.getOptimalNewCameraMatrix(self.cam_mat, self.dist_mat, (w, h), 1, (w, h))
        
        return self.cam_mat, self.dist_mat, self.optimal_mat

    def serialize(self):
        return {
                "checkerboard_size" : self._cb_size,
                "image_count"       : self._img_count,
                "cam_mat"           : self.cam_mat.tolist(),
                "dist_mat"          : self.dist_mat.tolist(),
                "optimal_mat"       : self.optimal_mat.tolist()
            }

    def deserialize(self):
        return None