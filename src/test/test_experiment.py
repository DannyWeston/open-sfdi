import numpy as np
import pytest

from pathlib import Path

from opensfdi.io.services import ExperimentService, ImageService

from test.video import FakeCamera, FakeFringeProjector

from opensfdi.experiment import FPExperiment
from opensfdi.phase_shifting import NStepPhaseShift
from opensfdi.phase_unwrapping import ReliabilityPhaseUnwrap

from opensfdi.profilometry import LinearInversePH, PolynomialPH
from opensfdi.utils import show_surface, show_heightmap


img_service = ImageService()
ex_service = ExperimentService()

def ask_folder():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    return Path(filedialog.askdirectory())

@pytest.mark.skip(reason="Not ready")
def test_calibration():
    # Declare paths for loading stuff
    img_path = ask_folder()

    count = 7
    phases = 12
    heights = np.linspace(0.0, 24.0, count, dtype=np.float64)
    imgs = []

    for i in range(count):
        group = []
        for j in range(phases):
            img = img_service.load_image(img_path / f"height{i}_phase{j+1}.jpg", greyscale=True)
            if img is None: raise Exception(f"Could not load image")
            
            group.append(img)

        imgs.append(np.array(group))

    imgs = np.array(imgs)

    # Get unwrapped phasemaps of loaded images
    print("Starting polynomial calibration...")

    ph_shift = NStepPhaseShift(steps=phases)
    ph_unwrap = ReliabilityPhaseUnwrap()

    z, _, h, w = imgs.shape
    phasemaps = np.empty(shape=(z, h, w))

    for i, height_imgs in enumerate(imgs):
        phasemaps[i] = ph_unwrap.unwrap(ph_shift.shift(height_imgs))

    # Perform calibration
    calib = PolynomialPH(degree=5)
    calib.calibrate(phasemaps, heights)

    # Load the calibration data
    calib_save_dir = ask_folder()
    ex_service.save_ph_calib(calib, calib_save_dir)

    print(f"Finished calibration")

def test_experiment():
    # Load the calibration data
    calib_dir = ask_folder()
    calib = ex_service.load_ph_calib(calib_dir)

    img_dir = ask_folder()

    # Load the measurement images
    phases = 12
    imgs = [img_service.load_image(img_dir / f"img{i}.jpg", greyscale=True) for i in range(phases * 2)]
    print("Finished loading measurement images")

    # Calculate phasemap difference for measurement + reference images
    camera = FakeCamera(imgs)
    projector = FakeFringeProjector()

    ph_shift = NStepPhaseShift(phases)
    ph_unwrap = ReliabilityPhaseUnwrap()

    heightmap = FPExperiment(camera, projector, ph_shift, ph_unwrap, calib).run()

    # Show the result
    show_heightmap(heightmap)