import unittest


import numpy as np


from opensfdi.fringes import FringeGenerator


class TestFringesCalibration(unittest.TestCase):

    def test_generation(self):

        width = 2048

        height = 2048


        freq = 32

        orientation = np.pi

        fringes = FringeGenerator.from_generator(width, height, freq, orientation)


        fringes.save([f'fringes{i}.jpg' for i in range(len(fringes))])


        self.assertTrue(True)