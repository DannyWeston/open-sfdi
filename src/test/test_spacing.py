import tkinter as tk
import numpy as np
import pytest
import matplotlib.pyplot as plt

from pathlib import Path
from tkinter import filedialog

from opensfdi import image, stereo
from opensfdi.devices import board
from opensfdi.phase import unwrap
from opensfdi.services import FileImageRepo, CameraFileRepo, ProjectorFileRepo, save_pointcloud


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

  camera = board.FileCamera(resolution=(1080, 1920), channels=1)
  projector = board.FakeProjector(resolution=(1140, 912), pixel_size=0.8, throw_ratio=1.0)
  calib_board = board.CircleBoard(circleSpacing=0.03, poiCount=(4, 13), inverted=True, staggered=True, areaHint=(2500, 13000))

  shifter = unwrap.NStepPhaseShift(phase_count=phase_count, shift_mask=shift_mask)
  unwrapper = unwrap.MultiFreqPhaseUnwrap(fringe_counts)
  calibrator = stereo.ZhangCharacteriser(calib_board)

  for spacing in spacings:
      # Set correct camera images
      spacing_path = exp_root / spacing
      img_repo = FileImageRepo(spacing_path, use_ext='.tif')
      camera.imgs = list(img_repo.GetBy("calibration", sorted=True))

      calibrator.Characterise(camera, projector, shifter, unwrapper, poseCount=orientations)

      # Save the experiment information and the calibrated camera / projector
      cam_repo = CameraFileRepo(spacing_path, overwrite=True)
      proj_repo = ProjectorFileRepo(spacing_path, overwrite=True)

      cam_repo.Add(camera, "camera")
      proj_repo.Add(projector, "projector")

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
  shifter = unwrap.NStepPhaseShift(phase_count, shift_mask=shift_mask)
  unwrapper = unwrap.MultiFreqPhaseUnwrap(num_stripes)
  reconstructor = stereo.StereoProfil()

  for spacing in spacings:
    path = exp_root / spacing


    cam_repo = CameraFileRepo(path, overwrite=True)
    camera: board.FileCamera = cam_repo.Get("camera")
    img_repo = FileImageRepo(path, use_ext='.tif', channels=camera.channels)

    proj_repo = ProjectorFileRepo(path, overwrite=True)
    projector: board.FakeProjector = proj_repo.Get("projector")

    for obj in objects:
      camera.imgs = list(img_repo.GetBy(f"{obj}_", sorted=True))

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