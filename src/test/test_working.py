import tkinter as tk
import numpy as np
import pytest
import matplotlib.pyplot as plt

from pathlib import Path
from tkinter import filedialog

from opensfdi import calibration, image
from opensfdi.devices import board
from opensfdi.phase import unwrap
from opensfdi.services import FileImageRepo, FileCameraConfigRepo, FileProjectorRepo, save_pointcloud

# Initialise tkinter for file browsing
# TODO: Change this to use paths etc
root = tk.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()
print("")

circle_spacings = [
  0.03,


  0.0225,
]

working_distances = [
  0.0,
  0.03,
  0.06,
  0.09,
  0.12,
  0.15,
  0.18,
]

@pytest.mark.skip(reason="Not ready")
def test_calibration():
  # Instantiate classes for calibration
  # Load the images to use in the calibration process

  # imgs = list(ex_service.get_by("calibration*", sorted=True))
  
  # TODO: Fix devices folder creation
  exp_root = Path(filedialog.askdirectory(title="Where is the folder for the resolution experiments?"))

  # Fringe projection & calibration parameters
  shift_mask = 0.03
  phase_count = 8
  fringe_counts = [1.0, 8.0, 64.0]
  orientations = 17

  resolutions = [
    (480, 270),
    (640, 360),
    (960, 540),
    (1440, 810),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
  ]

  projector = board.FakeProjector(resolution=(1140, 912), pixel_size=0.8, throw_ratio=1.0)
  shifter = unwrap.NStepPhaseShift(phase_count, shift_mask=shift_mask)
  unwrapper = unwrap.MultiFreqPhaseUnwrap(fringe_counts)

  for res in resolutions:
    res_path = exp_root / f"{res[0]}x{res[1]}"
    img_repo = FileImageRepo(res_path, fileExt='.tif')
    
    camera = board.FileCamera(resolution=res[::-1], channels=1, imgs=list(img_repo.GetBy("calibration", sorted=True)))

    area_min = (res[0] * res[1]) / 1000
    area_max = area_min * 4.5
    calib_board = board.CircleBoard(circleSpacing=0.03, poiCount=(4, 13), inverted=True, staggered=True, areaHint=(area_min, area_max))

    calibrator = calibration.StereoCalibrator(calib_board)
    calibrator.Calibrate(camera, projector, shifter, unwrapper, imageCount=orientations)

    # Save the experiment information and the calibrated camera / projector
    FileCameraConfigRepo(res_path, overwrite=True).Add(camera, "camera")
    FileProjectorRepo(res_path, overwrite=True).Add(projector, "projector")

@pytest.mark.skip(reason="Not ready")
def test_measurement():
  # Fringe projection settings
  cb_size     = (4, 13)
  square_size = 0.03
  shift_mask  = 0.05
  phase_count = 8
  num_stripes = [1.0, 8.0, 64.0]

  objects = [
    "Hand",
    "Icosphere",
    "Monkey",
    "Donut",
  ]

  # Load projector and camera with imgs
  calib_path = Path(filedialog.askdirectory(title="Where is the folder for the optical devices?"))
  
  cam_repo = FileCameraConfigRepo(calib_path, overwrite=True)
  camera: board.FileCamera = cam_repo.Get("camera")
  img_repo = FileImageRepo(calib_path, fileExt='.tif', channels=camera.channels)

  proj_repo = FileProjectorRepo(calib_path, overwrite=True)
  projector: board.FakeProjector = proj_repo.Get("projector")

  # Phase related stuff
  shifter = unwrap.NStepPhaseShift(phase_count, shift_mask=shift_mask)
  unwrapper = unwrap.MultiFreqPhaseUnwrap(num_stripes)
  reconstructor = calibration.StereoProfil()

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

    save_pointcloud(calib_path / f"{obj}.ply", pc)

    import open3d as o3d
    pcd_load = o3d.io.read_point_cloud(calib_path / f"{obj}.ply")
    o3d.visualization.draw_geometries([pcd_load])

  # TODO: Consider rotation, save as .ply
   
  #pc = pointcloud.rotate_pointcloud(pc, np.pi / 2.0, 0.0, np.pi)

  # From Blender coords to export stl with:
  # Up: Y
  # Forward: -Z