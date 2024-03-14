import os
import logging
import numpy as np

from datetime import datetime

from sfdi.io.repositories import ImageRepo, FileImageRepo, CalibrationRepo, BinRepo, BinCalibrationRepo
from sfdi.definitions import RESULTS_DIR, CALIBRATION_DIR

class CalibrationService():
    def __init__(self, data_repo:CalibrationRepo = None):       
        self._logger = logging.getLogger('sfdi')

        self._data_repo = data_repo

        if self._data_repo is None: # Default to bin repo
            output = os.path.join(CALIBRATION_DIR, 'calibration.json')
            self._data_repo = BinCalibrationRepo(output)
            
            self._logger.debug(f'Using calibration data file {output}')

    def save_calibrations(self, gamma_calib=None, lens_calib=None, proj_calib=None):
        updated = False
        if gamma_calib:
            self._data_repo.add_gamma(gamma_calib)
            updated = True
            
        if lens_calib: 
            self._data_repo.add_lens(lens_calib)
            updated = True
            
        if proj_calib: 
            self._data_repo.add_proj(proj_calib)
            updated = True
        
        if updated: 
            self._data_repo.commit()

    def load_calibrations(self, cam_name, proj_name):
        return self._data_repo.load_gamma(cam_name), self._data_repo.load_lens(cam_name), self._data_repo.load_proj(proj_name)

class ResultService():
    def __init__(self, data_repo:BinRepo, image_repo:ImageRepo):
        self._logger = logging.getLogger('sfdi')
        
        self._data_repo = data_repo
        self._image_repo = image_repo

    @staticmethod
    def default(directory=None):
        if directory is None: directory = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
        loc = os.path.join(RESULTS_DIR, directory)
        
        if not os.path.exists(loc): os.mkdir(loc, 0o770)
        
        data_out = os.path.join(loc, 'results.bin')
        
        return ResultService(BinRepo(data_out), FileImageRepo(loc))

    def save_data(self, data=None, fringes=None, imgs=None, ref_imgs=None):
        if data is not None:
            self._data_repo.add_bin(data)
            self._data_repo.commit()

        updated = False
            
        if fringes is not None:
            updated = True
            for i, img in enumerate(fringes): 
                self._image_repo.add_image(img, f'fringes{i}.jpg') 
            
        if imgs is not None:
            updated = True
            for cam_i, xs in enumerate(imgs):
                for i, img in enumerate(xs):
                    self._image_repo.add_image(img, f'cam{cam_i}_img{i}.jpg')
                    
        if ref_imgs is not None:
            updated = True
            for cam_i, xs in enumerate(ref_imgs):
                for i, img in enumerate(xs):
                    self._image_repo.add_image(img, f'cam{cam_i}_refimg{i}.jpg')

        if updated: self._image_repo.commit()

    def load_data(self):
        data = self._data_repo.load_bin()
        cam_count = len(data["cameras"].keys())
        phases = data["phases"]
        
        imgs = np.empty((cam_count, phases), dtype=np.ndarray)
        ref_imgs = np.empty((cam_count, phases), dtype=np.ndarray)
        for cam_i in range(cam_count):
            for i in range(phases):
                imgs[cam_i][i] = self._image_repo.load_image(f'cam{cam_i}_img{i}.jpg')
                ref_imgs[cam_i][i] = self._image_repo.load_image(f'cam{cam_i}_refimg{i}.jpg')

        return ref_imgs, imgs, data