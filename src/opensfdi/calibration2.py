import logging
import numpy as np

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

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