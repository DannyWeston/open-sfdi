import numpy as np
import pytest

from pathlib import Path

import opensfdi.phase as phase
import opensfdi.profilometry as prof

from opensfdi.services import ExperimentService, ImageService, FileProfRepo, FileImageRepo
from opensfdi.experiment import FPExperiment
from opensfdi.utils import show_surface, show_heightmap

from test.video import FakeCamera, FakeFringeProjector

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

def test_calibration():
    # Declare paths for loading stuff
    img_repo = FileImageRepo(storage_dir=Path(filedialog.askdirectory(title="Where to load images from:")))
    img_service = ImageService(img_repo)

    count = 7
    phases = 12
    heights = np.linspace(0.0, 24.0, count, dtype=np.float64)
    imgs = []

    # Load all of the test images

    for i in range(count):
        for j in range(phases):
            img = img_service.load_image(f"height{i}_phase{j+1}.jpg")
            img = np.mean(img, axis=2)

            if img is None: raise Exception(f"Could not load image")
        
            imgs.append(img)

    print("Starting polynomial calibration...")

    # Calculate phasemap difference for measurement + reference images
    ph_shift = phase.NStepPhaseShift(steps=phases)
    ph_unwrap = phase.ReliabilityPhaseUnwrap()
    calib = prof.PolynomialProf(name="polynomial1", degree=5)
    experiment = FPExperiment(FakeCamera(imgs), FakeFringeProjector(), ph_shift, ph_unwrap, calib)

    def f(height): print(f"Height: {height} mm")

    experiment.on_height_measurement(f)
    experiment.calibrate(heights)

    # Save the calibration data
    repo = FileProfRepo(storage_dir=Path(filedialog.askdirectory(title="Where to save calibrations to")))
    ex_service = ExperimentService(repo)
    ex_service.save_calib(calib)

    print(f"Finished calibration")

@pytest.mark.skip(reason="Not ready")
def test_experiment():
    ex_service = ExperimentService(FileProfRepo(filedialog.askdirectory(title="What calibration directory to use?")))
    prof = ex_service.load_calib(filedialog.askopenfilename(title="What calibration to use?"))

    # Declare paths for loading stuff
    img_service = ImageService(FileImageRepo(filedialog.askdirectory(title="Where to load measurement images from:")))

    # Load the measurement images
    phase_count = 12
    imgs = np.array([img_service.load_image(f"img{i}.jpg") for i in range(phase_count * 2)])
    imgs = np.mean(imgs, axis=3)
 # Turn to greyscale
    print("Finished loading measurement images")

    # Calculate phasemap difference for measurement + reference images
    camera = FakeCamera(imgs.tolist())
    projector = FakeFringeProjector()

    ph_shift = phase.NStepPhaseShift(phase_count)
    ph_unwrap = phase.ReliabilityPhaseUnwrap()

    heightmap = FPExperiment(camera, projector, ph_shift, ph_unwrap, prof).run()

    # Show the result
    show_heightmap(heightmap)