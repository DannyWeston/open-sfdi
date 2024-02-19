import unittest

import numpy as np
import os
import cv2

from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval

from sfdi.fringes import Fringes
from sfdi.calibration.gamma_calibration import GammaCalibration

from sfdi.definitions import CALIBRATION_DIR

class TestGammaCalibration(unittest.TestCase):
    def test_generation(self):
        intensity_count = 32

        intensities = np.linspace(0, 255, intensity_count)

        # TODO: Create some fake calibration images
        form = [3.0, 2.0, 1.0] # 3x^2 + 2x + 1
        
        a = polyval([1, 2], form)
        
        # captured_imgs = []
        
        # coeffs, values = GammaCalibration.calculate_curve(captured_imgs, intensities, delta=0.25, order=5)

        # print(f'Imported: {CALIBRATION_DIR}')
        # GammaCalibration.save_calibration(coeffs, values) # Save the results

        # fringes = Fringes.from_generator(2048, 2048, 64)

        # # TODO: Tidy up code
        # fringes.images = [
        #     GammaCalibration.apply_correction(pattern, coeffs, values.min(), values.max()) 
        #     for pattern in fringes
        # ]

        self.assertTrue(True)

    def test_load_calibration(self):
        test_coeffs = np.array([0.0, 1.0, 2.0])
        test_values = np.array([3.0, 4.0, 5.0])

        outname = 'test_gamma_calibration.json'

        GammaCalibration.save_calibration(test_coeffs, test_values, outname)

        coeffs, values = GammaCalibration.load_calibration(outname)

        self.assertListEqual(test_coeffs.tolist(), coeffs.tolist())
        self.assertListEqual(test_values.tolist(), values.tolist())