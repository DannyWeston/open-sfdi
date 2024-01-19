import unittest

import numpy as np

from sfdi.generation.fringes import Fringes

class TestFringesCalibration(unittest.TestCase):
    def test_generation(self):
        width = 2048
        height = 2048

        freq = 32
        orientation = np.pi
        fringes = Fringes.from_generator(width, height, freq, orientation)

        fringes.save([f'fringes{i}.jpg' for i in range(len(fringes))])

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()