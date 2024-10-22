import numpy as np
import cv2
import os

from numpy.polynomial import polynomial as P
from pathlib import Path

from skimage.restoration import unwrap_phase

from opensfdi import wrapped_phase, show_surface, show_phasemap
from opensfdi.profilometry import poly_phase_height

if __name__ == "__main__":
    # Load the images from disk
    input_dir = Path('C:\\Users\\danie\\Desktop\\Results\\Test1')

    # Load the calibration images
    calib_dir = input_dir / "Calibration"

    calib_file = calib_dir / "calib.npy"
    
    if os.path.exists(calib_file):
        coeffs = np.load(calib_file)
    else:
        c_count = 20
        c_phases = 4
        c_dists = np.linspace(0.0, 20.0, c_count, dtype=np.float64)
        c_images = []

        for i in range(c_count):
            group = []
            for j in range(c_phases):
                img = cv2.imread(calib_dir / f"{i+1}_phase{j}.jpg", cv2.IMREAD_GRAYSCALE)
                group.append(img)

            c_images.append(np.array(group))

        c_images = np.array(c_images)

        print("Starting polynomial calibration...")

        coeffs = poly_phase_height(c_images, c_dists, 3)

        print("Saving calibration data...")

        with open(calib_file, "wb") as out_file:
            np.save(out_file, coeffs)

        print(f"Finished calibration (saved to {calib_file})")

    # Now have the coefficients for height, so can use for construction
    # Load the measurement images
    meas_dir = input_dir / "Measurement"
    phases = 12

    ref_imgs = []
    meas_imgs = []

    for i in range(phases):
        ref_img = cv2.imread(meas_dir / f"ref_{i}.jpg", cv2.IMREAD_GRAYSCALE)
        ref_imgs.append(ref_img)

        meas_img = cv2.imread(meas_dir / f"measurement_{i}.jpg", cv2.IMREAD_GRAYSCALE)
        meas_imgs.append(meas_img)
    
    print("Finished loading measurement images")


    # Calculate phasemap difference for measurement + reference images

    ref_phase = wrapped_phase(ref_imgs)
    ref_phase = unwrap_phase(ref_phase)

    meas_phase = wrapped_phase(meas_imgs)
    meas_phase = unwrap_phase(meas_phase)

    phase_diff = meas_phase - ref_phase


    # Apply polynomial fit to obtained phase difference in order to obtain depth

    h, w = phase_diff.shape
    height = np.zeros_like(phase_diff)

    for y in range(h):
        for x in range(w):
            height[y, x] = P.polyval(phase_diff[y, x], coeffs[:, y, x])

    show_surface(height)