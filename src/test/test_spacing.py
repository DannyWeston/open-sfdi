import tkinter as tk
import numpy as np
import pytest
import matplotlib.pyplot as plt

from pathlib import Path
from tkinter import filedialog

from opensfdi import profilometry, phase, image, devices
from opensfdi.services import FileImageRepo, FileCameraRepo, FileProjectorRepo, save_pointcloud


from cv2 import ocl
print(f"OpenCL supported: {ocl.haveOpenCL()}")
exp_root = Path(filedialog.askdirectory(title="Where is the folder for the spacings?"))

spacings = [
  # "+0", 
  # "+5", 
  # "+10", 
  # "+15", 
  # "+20",
  # "+25",
  "+30"
]

@pytest.mark.skip(reason="Not ready")
def test_calibration():
  # Instantiate classes for calibration

  # Fringe projection & calibration parameters
  shift_mask = 0.03
  phase_count = 8
  fringe_counts = [1.0, 8.0, 64.0]
  orientations = 17

  camera = devices.FileCamera(resolution=(1080, 1920), channels=1)
  projector = devices.FakeProjector(resolution=(1140, 912), pixel_size=0.8, throw_ratio=1.0)
  calib_board = devices.CircleBoard(circle_spacing=0.03, poi_count=(4, 13), inverted=True, staggered=True, area_hint=(2500, 13000))

  shifter = phase.NStepPhaseShift(phase_count=phase_count, shift_mask=shift_mask)
  unwrapper = phase.MultiFreqPhaseUnwrap(fringe_counts)
  calibrator = profilometry.StereoCalibrator(calib_board)

  for spacing in spacings:
      # Set correct camera images
      spacing_path = exp_root / spacing
      img_repo = FileImageRepo(spacing_path, file_ext='.tif')
      camera.imgs = list(img_repo.get_by("calibration", sorted=True))

      calibrator.calibrate(camera, projector, shifter, unwrapper, num_imgs=orientations)

      # Save the experiment information and the calibrated camera / projector
      cam_repo = FileCameraRepo(spacing_path, overwrite=True)
      proj_repo = FileProjectorRepo(spacing_path, overwrite=True)

      cam_repo.add(camera, "camera")
      proj_repo.add(projector, "projector")

      # TODO: Save experiment results
  
# @pytest.mark.skip(reason="Not ready")
def test_measurement():
  # Fringe projection settings
  shift_mask  = 0.03
  phase_count = 8
  num_stripes = [1.0, 8.0, 64.0]

  objects = [
    "Hand",
    "Icosphere",
    "Monkey",
    "Donut",
  ]

  # Load projector and camera with imgs
  
  # Phase related stuff
  shifter = phase.NStepPhaseShift(phase_count, shift_mask=shift_mask)
  unwrapper = phase.MultiFreqPhaseUnwrap(num_stripes)
  reconstructor = profilometry.StereoProfil()

  for spacing in spacings:
    path = exp_root / spacing


    cam_repo = FileCameraRepo(path, overwrite=True)
    camera: devices.FileCamera = cam_repo.get("camera")
    img_repo = FileImageRepo(path, file_ext='.tif', channels=camera.channels)

    proj_repo = FileProjectorRepo(path, overwrite=True)
    projector: devices.FakeProjector = proj_repo.get("projector")

    for obj in objects:
      camera.imgs = list(img_repo.get_by(f"{obj}_", sorted=True))

      pc, _ = reconstructor.reconstruct(camera, projector, shifter, unwrapper, num_stripes[-1])

      # Positioned at camera coordinate frame origin (0, 0, 0)
      # offset_x, offset_y = profilometry.checkerboard_centre(cb_size, square_size)
      # pc[:, 0] -= offset_x
      # pc[:, 1] -= offset_y

      # Rotate by 90 degrees to align with OpenGL coordinate frame
      r_x = (np.pi / 2.0) #+ ((15.0 / 180.0) * np.pi)
      R_x = np.array([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)], [0, np.sin(r_x), np.cos(r_x)]])
      # pc = (R_x @ pc.T).T

      save_pointcloud(path / f"{obj}.ply", pc)

    import open3d as o3d
    pcd_load = o3d.io.read_point_cloud(path / "Hand.ply")
    o3d.visualization.draw_geometries([pcd_load])

    # TODO: Save experiment results
  

  # TODO: Consider rotation, save as .ply
   
  #pc = pointcloud.rotate_pointcloud(pc, np.pi / 2.0, 0.0, np.pi)

  # From Blender coords to export stl with:
  # Up: Y
  # Forward: -Z