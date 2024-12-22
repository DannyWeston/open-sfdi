import numpy as np
import pytest
import tkinter as tk

import opensfdi.phase as phase
import opensfdi.profilometry as prof

from pathlib import Path
from tkinter import filedialog

from opensfdi.services import ExperimentService, FileExperimentRepo, FileImageRepo
from opensfdi.experiment import Experiment
from test.video import FakeCamera, FakeFringeProjector

root = tk.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()

@pytest.mark.skip(reason="Not ready")
def test_ph_calib():
    exp_path = Path(filedialog.askdirectory(title="What experiment directory to use?"))
    ex_service = ExperimentService(FileExperimentRepo(exp_path), FileImageRepo(exp_path))

    imgs = ex_service.load_imgs("calibration_*") # Load the calibration images
    heights = imgs.shape[0]

    # Make the mock devices
    fake_camera = FakeCamera(np.mean(imgs, axis=3))
    fake_proj = FakeFringeProjector()

    # Make the experiment
    ph_shift = phase.NStepPhaseShift(phase_count=len(imgs) / 2)
    ph_unwrap = phase.ReliabilityPhaseUnwrap()
    profil = prof.PolynomialProf()

    exp = Experiment("testexp", profil, ph_shift, ph_unwrap)
    height_imgs = [exp.get_imgs(fake_camera, fake_proj) for _ in range(heights)]

    # Add height callback
    def on_h(height): print(f"Height: {height} mm")
    exp.on_height_measurement(on_h)

    # Run the calibration
    exp.calibrate(height_imgs, np.linspace(0.0, 24.0, heights, dtype=np.float64))

    # Save the calibration
    ex_service.save_experiment(exp)

@pytest.mark.skip(reason="Not ready")
def test_run_experiment():
    exp_path = Path(filedialog.askdirectory(title="What experiment directory to use?"))
    ex_service = ExperimentService(FileExperimentRepo(exp_path), FileImageRepo(exp_path))

    imgs = ex_service.load_imgs("measurement_*") # Load the measurement images
    imgs = np.mean(imgs, axis=3) # Convert to greyscale

    # Make the mock devices
    fake_camera = FakeCamera(np.mean(imgs, axis=3))
    fake_proj = FakeFringeProjector()

    exp = ex_service.load_experiment("testexp")

    imgs = exp.get_imgs(fake_camera, fake_proj)
    heightmap = exp.heightmap(imgs)

    # TODO: View heightmap

def test_save_experiment():
    exp_path = Path(filedialog.askdirectory(title="What experiment directory to use?"))
    ex_service = ExperimentService(FileExperimentRepo(exp_path), FileImageRepo(exp_path))

    profil = prof.PolynomialProf(data=np.linspace(0.0, 10.0, 32))
    ph_shift = phase.NStepPhaseShift(phase_count=5)
    ph_unwrap = phase.ReliabilityPhaseUnwrap()

    print()
    exp_name = input("What name to use for the experiment?\n")
    experiment = Experiment(exp_name, profil, ph_shift, ph_unwrap)

    # Save the experiment
    ex_service.save_experiment(experiment)

def test_load_experiment():   
    exp_path = Path(filedialog.askdirectory(title="What experiment directory to use?"))
    ex_service = ExperimentService(FileExperimentRepo(exp_path), FileImageRepo(exp_path))

    print()
    print(ex_service.get_exp_list())

    exp_name = input("What experiment do you want to load?\n")
    exp = ex_service.load_experiment(exp_name)

    print(exp)
    print(exp.profil.data)