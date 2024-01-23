import numpy as np
import os
import json

from sfdi.definitions import CALIBRATION_DIR

class GammaCalibration:
    def __init__(self):
        pass

    def rgb2grey(img):
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def calculate_curve(imgs, interp, delta=0.25, crop_size=0.25, order=5):
        if len(imgs) != interp.size:
            raise Exception("Must have as many images as interpolation values")
        
        cap_width, cap_height, _ = imgs[0].shape

        # Calculate region of interest values
        roi = int(cap_width * crop_size)
        mid_height = int(cap_height / 2)
        mid_width = int(cap_width / 2)
        rows = [x + mid_height for x in range(-roi, roi)]
        cols = [x + mid_width for x in range(-roi, roi)]

        # Calculate average pixel value for each image
        averages = [np.mean(x[cols, rows]) for x in imgs]

        # Normalise to 0-255
        averages = averages / np.max(averages) * 255 

        # Find sfirst observable change of values for averages (left and right sides) i.e >= delta
        s, f = GammaCalibration._detectable_indices(averages, delta)

        vis_averages = averages[s:f+1]
        vis_intensities = interp[s:f+1]

        coeffs = np.polyfit(vis_averages, vis_intensities, order)

        # Plot results

        # plt.plot(vis_averages, vis_intensities, 'o')

        # trendpoly = np.poly1d(coeffs)

        # plt.title('Gamma Calibration Curve')

        # plt.plot(vis_averages, trendpoly(vis_averages))

        return coeffs, vis_intensities, 

    def apply_correction(img, coeffs, x1=0, x2=255):
        if img.ndim == 2: # Assume already greyscale
            corrected_img = img.astype(np.double)

        elif img.ndim == 3: # Assume RGB so need to convert to greyscale
            corrected_img = GammaCalibration.rgb2grey(img.astype(np.double))

        else: raise Exception("Image does not have correct dimensions")

        poly_func = np.poly1d(coeffs)

        corrected_img = poly_func(corrected_img)

        corrected_img[corrected_img < x1] = x1
        corrected_img[corrected_img > x2] = x2

        return corrected_img.astype(np.uint8)

    def save_calibration(coeffs, values, name=None):
        data = {
            'coeffs': coeffs.tolist(),
            'values': values.tolist()
        }

        with open(os.path.join(CALIBRATION_DIR, name if name else 'gamma_calibration.json'), 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def load_calibration(name=None):
        with open(os.path.join(CALIBRATION_DIR, name if name else 'gamma_calibration.json'), 'r') as infile:
            raw_json = json.load(infile)
            return np.array(raw_json['coeffs']), np.array(raw_json['values'])
    
    def _detectable(xs, delta):
        start = finish = None

        for i in range(1, len(xs) - 1):
            x1 = xs[i - 1]
            x2 = xs[i]

            y1 = xs[:(-i) - 1]
            y2 = xs[:-i]

            if not start and abs(x1 - x2) >= delta:
                start = i

            if not finish and abs(y1 - y2) >= delta:
                finish = len(xs) - i - 1

        return xs[start:finish + 1]

    def _detectable_indices(values, delta):
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