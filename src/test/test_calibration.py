import tkinter as tk
import numpy as np
import pytest

from pathlib import Path
from tkinter import filedialog

from opensfdi import profilometry, phase
from opensfdi.fringe_projection import FringeProjection
from opensfdi.services import ExperimentService, FileExperimentRepo, FileImageRepo

from .video import FakeCamera, FakeProjector
from opensfdi import image

# Initialise tkinter for file browsing
# TODO: Change this to use paths etc
root = tk.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()
repo_path = Path(filedialog.askdirectory(title="What directory for repo to use?"))
ex_service = ExperimentService(FileExperimentRepo(repo_path), FileImageRepo(repo_path, greyscale=True))
print("")

reconst = None

@pytest.mark.skip(reason="Not ready")
def test_pointcloud():
  profilometry.show_pointcloud(np.random.random((100, 100, 3)), 'Test')


@pytest.mark.skip(reason="Not ready")
def test_calibration():
  # Instantiate classes for calibration
  # Load the images to use in the calibration process

  #imgs = list(ex_service.get_by("calibration*", sorted=True))
  global ex_service
  imgs = list(ex_service.get_by(f"calibration", sorted=True))
  # imgs = list(ex_service.load_img(f"calibration{str(i).zfill(5)}") for i in range(1800))

  camera = FakeCamera(imgs, resolution=(1080, 1920), channels=1)
  projector = FakeProjector(resolution=(1080, 1920))

  phase_count = 8
  phases = [1.0, 15.0, 180.0]
  num_imgs = 36

  calib = profilometry.StereoCalibrator(
    phase.NStepPhaseShift(phase_count),
    phase.MultiFreqPhaseUnwrap(phases)
  )

  assert len(imgs) == len(phases) * phase_count * num_imgs
  print(f"{len(imgs)} images loaded")

  global reconst
  reconst = calib.calibrate(camera, projector, num_imgs)

  ex_service.save_experiment(reconst, "experiment1")

  print("Calibration finished")

  # Undistort images
  # Determine whether to use black areas or not - alpha = % of filled-in pixels for undistorted image
  # self.optimal_mat, roi = cv2.getOptimalNewCameraMatrix(self.cam_mat, self.dist_mat, (w, h), 1, (w, h))

  # # Save the calibration
  # ex_service.save_experiment(exp)


@pytest.mark.skip(reason="Not ready")
def test_gamma():
  global ex_service
  imgs = list(ex_service.get_by(f"gamma*", sorted=True))

  measured_gammas = np.array([image.calc_gamma(img.raw_data) for img in imgs])
  expected_gammas = np.linspace(0.0, 1.0, len(measured_gammas), dtype=np.float32)

  image.show_scatter([expected_gammas], [measured_gammas])


@pytest.mark.skip(reason="Not ready")
def test_vignette():
  global ex_service
  img = ex_service.load_img("gamma1").raw_data

  diff = (np.ones_like(img, dtype=np.float32) * img.max()) - img
  
  image.show_image(diff, "Vignetting amount")
  image.show_image(img + diff, "Vignetting fixed")


@pytest.mark.skip(reason="Not ready")
def test_measurement():
  # Load the images to use in the calibration process
  global ex_service
  imgs = list(ex_service.get_by(f"measurement", sorted=True))

  #unwrapper = phase.TemporalPhaseUnwrap(phase_count=3, spatial_freqs=[1.0, 16.0, 64.0])

  reconst = ex_service.load_experiment("experiment1")

  # fringe_proj = FringeProjection(
  #   phase.NStepPhaseShift(phase_count=8),
  #   phase.MultiFreqPhaseUnwrap(np.array([1.0, 15.0, 180.0]))
  # )

  camera = FakeCamera(imgs, resolution=(1080, 1920))
  projector = FakeProjector(resolution=(1080, 1920))

  reconst.reconstuct(imgs)