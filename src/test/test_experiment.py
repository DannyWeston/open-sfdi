import numpy as np
import cv2
import os

from pathlib import Path
from test.video import FakeCamera, FakeFringeProjector

from opensfdi.experiment import FPExperiment
from opensfdi.phase_shifting import NStepPhaseShift
from opensfdi.phase_unwrapping import ReliabilityPhaseUnwrap

from opensfdi.profilometry.phase import PolynomialPH
from opensfdi.profilometry import show_surface, show_heightmap

def test_experiment():
    # Declare paths for loading stuff
    input_dir = Path('C:\\Users\\danie\\Desktop\\Results\\Test1')

    calib_dir = input_dir / "Calibration"

    calib_file = calib_dir / "calib.npy"
    

    # Load coeffs if they already exist

    if os.path.exists(calib_file):
        coeffs = np.load(calib_file)
        poly_ph = PolynomialPH(coeffs)

    else:
        count = 20
        phases = 4
        heights = np.linspace(0.0, 20.0, count, dtype=np.float64)
        imgs = []

        for i in range(count):
            group = []
            for j in range(phases):
                img = cv2.imread(calib_dir / f"{i+1}_phase{j}.jpg", cv2.IMREAD_GRAYSCALE)
                group.append(img)

            imgs.append(np.array(group))

        imgs = np.array(imgs)

        print("Starting polynomial calibration...")

        # Get unwrapped phasemaps of loaded images
        ph_shift = NStepPhaseShift(steps=phases)
        ph_unwrap = ReliabilityPhaseUnwrap()

        phasemaps = []
        for height_imgs in imgs:
            shifted = ph_shift.shift(height_imgs)
            phasemaps.append(ph_unwrap.unwrap(shifted))

        # Perform calibration
        poly_ph = PolynomialPH()
        poly_ph.calibrate(phasemaps, heights, 5) # 5th degree polynomial
        
        # Save the data
        poly_ph.save_data(calib_file)
        print(f"Finished calibration (saved to {calib_file})")


    # Load the measurement images

    meas_dir = input_dir / "Measurement"
    phases = 12

    imgs = []
    temp = []

    temp = [cv2.imread(meas_dir / f"ref_{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(phases)]
    imgs.append(temp)

    temp = [cv2.imread(meas_dir / f"measurement_{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(phases)]
    imgs.append(temp)

    print("Finished loading measurement images")

    # Calculate phasemap difference for measurement + reference images

    camera = FakeCamera(imgs)
    projector = FakeFringeProjector()

    ph_shift = NStepPhaseShift()
    ph_unwrap = ReliabilityPhaseUnwrap()

    experiment = FPExperiment(camera, projector, ph_shift, ph_unwrap, poly_ph)
    heightmap = experiment.run()

    # Show the result
    show_surface(heightmap)