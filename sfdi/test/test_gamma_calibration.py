import unittest

import numpy as np
import os
import cv2

from sfdi.generation.fringes import Fringes
from sfdi.calibration.gamma_calibration import GammaCalibration

class TestGammaCalibration(unittest.TestCase):
    def test_generation(self):
        intensity_count = 32

        intensities = np.linspace(0, 255, intensity_count)

        #img_in_dir = 'C:\\OneDrive\\OneDrive - The University of Nottingham\\university\\phd\\year1\\Code\\BlenderFPP\\Images\\gamma_calibration\\'
        img_in_dir = 'C:\\Users\\psydw2\\OneDrive - The University of Nottingham\\university\\phd\\year1\\Code\\BlenderFPP\\Images\\gamma_calibration'

        captured_imgs = []
        for i in range(len(intensities)):
            temp = cv2.imread(os.path.join(img_in_dir, f'result{i}.jpg')).astype(np.uint8)
            captured_imgs.append(temp)

        coeffs, values = GammaCalibration.calculate_curve(captured_imgs, intensities, delta=0.25, order=5)

        GammaCalibration.save_calibration(coeffs, values) # Save the results

        fringes = Fringes.from_generator(2048, 2048, 64)

        # TODO: Tidy up code
        fringes.images = [
            GammaCalibration.apply_correction(pattern, coeffs, values.min(), values.max()) 
            for pattern in fringes
        ]

        self.assertTrue(True)

    def test_load_calibration(self):
        test_coeffs = np.array([0.0, 1.0, 2.0])
        test_values = np.array([3.0, 4.0, 5.0])

        outname = 'test_gamma_calibration.json'

        GammaCalibration.save_calibration(test_coeffs, test_values, outname)

        coeffs, values = GammaCalibration.load_calibration(outname)

        self.assertListEqual(test_coeffs.tolist(), coeffs.tolist())
        self.assertListEqual(test_values.tolist(), values.tolist())

if __name__ == '__main__':
    unittest.main()