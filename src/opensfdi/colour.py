import matplotlib.pyplot as plt

import numpy as np

from enum import Enum
from scipy.optimize import curve_fit

from . import utils, image

def DetectableIndices(values, delta):
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

def RegionBasedIntensities(imgs, regionsX=3, regionsY=3):
    xp = utils.ProcessingContext().xp

    h, w, *_ = imgs[0].shape

    totalRegions = regionsY * regionsX

    actual = xp.empty(shape=(totalRegions, len(imgs)))

    for i in range(len(imgs)):
        for x in range(regionsX):
            for y in range(regionsY):
                x1 = int((x / regionsX) * (w - 1))
                x2 = int(((x + 1) / regionsX) * (w - 1))

                y1 = int((y / regionsY) * (h - 1))
                y2 = int(((y + 1) / regionsY) * (h - 1))
                
                actual[y * regionsX + x, i] = xp.mean(imgs[i, y1:y2, x1:x2])

    return actual

class GammaCorrector(utils.SerialisableMixin):
    def __init__(self, a, gamma, b):
        self._a = a
        self._gamma = gamma
        self._b = b

    def apply(self, img):
        # Apply correction: (img - b) / a) ^ gamma
        xp = utils.ProcessingContext().xp

        xp.maximum(((img - self.b) / self.a), 0, out=img)

        xp.power(img, self.gamma, out=img)

        xp.clip(img, 0, 1, out=img)
        
        return img

    @property
    def gamma(self):
        return self._gamma
    
    @property
    def a(self):
        return self._a
    
    @property
    def b(self):
        return self._b

    def __str__(self):
        return f"{self.a:.2f} * x^{self.gamma:.2f} + {self.b:.2f}"

class GammaCalibMethod(Enum):
    MEAN = 0

class GammaCalibrator:
    def __init__(self):
        pass

    def ObtainValue(self, img: image.Image, method=GammaCalibMethod.MEAN, crop=(0.25, 0.25, 0.75, 0.75)):
        xp = utils.ProcessingContext().xp

        img_data = img.raw_data

        h, w, *_ = img_data.shape

        # Setup region of interest on images
        if crop:
            start_x = int(w * crop[0])
            start_y = int(h * crop[1])

            end_x = int(w * crop[2])
            end_y = int(h * crop[3])

            img_data = img_data[start_y:end_y, start_x:end_x]

        # Determine which method to use for calculating a per-image value
        if method == GammaCalibMethod.MEAN:
            actual = xp.mean(img_data.flatten())

        return actual

    def Calculate(self, expected, actual, initial_guess=None) -> GammaCorrector:
        # Theres no need to run this on the GPU
        with utils.ProcessingContext.UseGPU(False):
            xp = utils.ProcessingContext().xp

            def f_x(x, a, gamma, b):
                return a * (x ** (1.0 / gamma)) + b
            
            # Initial guess: gamma=2.2, gain=1, offset=0
            if initial_guess is None:
                initial_guess = xp.asarray([1.0, 2.2, 0.0]) # a, gamma, b
            
            # Fit the curve (ToNumpy to fetch values from GPU if they are on there)
            # Note: the variables are bounded to remove divide by zero err:
            # 0.01 <= a <= inf
            # 0.1 <= gamma <= 5.0
            # -inf <= b <= inf

            params, covariance = curve_fit(
                f_x, expected, utils.ToContext(xp, actual), 
                bounds=([0.01, 0.1, -xp.inf], [xp.inf, 5.0, xp.inf]),
                p0=initial_guess
            )
            
            (a, gamma, b) = params
            
            return GammaCorrector(a, gamma, b)

    def Plot(self, expected, actual):
        # Plot the graph
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.set_title('Gamma Matching')

        virtual = actual * 2
        real = 2.5 * np.power(actual, (1.0/2.2)) - 0.2

        ax.scatter(expected, real, color='r')
        ax.scatter(expected, virtual, color='b')

        ax.legend(['Real', 'Virtual'])

        # ax.errorbar(xs+0.1, ys, yerr=xp.asarray([xp.std(camReprojErrs), xp.std(projReprojErrs), 0.01]), fmt="o", color='black', capsize=3)

        # ax.set_xticks(xs)
        ax.set_xlabel('Expected Intensity')
        ax.set_ylabel("Measured Intensity")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.02, 1.02)

        fig.tight_layout()

        plt.show()