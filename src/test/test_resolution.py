import tkinter as tk
import numpy as np
import pytest

from pathlib import Path
from tkinter import filedialog

from opensfdi import calibration as calib, reconstruction as recon
from opensfdi.devices import board, camera, projector
from opensfdi.phase import unwrap, shift
from opensfdi.services import FileImageRepo, FileCameraConfigRepo, FileProjectorRepo, save_pointcloud

from . import utils

# Initialise tkinter for file browsing
# TODO: Change this to use paths etc
root = tk.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()
print("")

resolutions = [
    # (480, 270),
    # (640, 360),
    # (960, 540),
    # (1440, 810),
    (1920, 1080),
    # (2560, 1440),
    # (3840, 2160),
]

# @pytest.mark.skip(reason="Not ready")
def test_calibration():
  # Instantiate classes for calibration
  # Load the images to use in the calibration process

  # imgs = list(ex_service.get_by("calibration*", sorted=True))
  
  # TODO: Fix devices folder creation
  expRoot = Path(filedialog.askdirectory(title="Where is the folder for the resolution experiments?"))

  # Fringe projection & calibration parameters
  shifterMask = 0.03
  phaseCount = 8
  stripeCounts = [1.0, 8.0, 64.0]
  boardPoses = 17

  proj = utils.FakeFPProjector(projector.ProjectorConfig(
    resolution=(1140, 912), channels=1, 
    throwRatio=1.0, pixelSize=0.8)
  )

  shifter = shift.NStepPhaseShift(phaseCount, mask=shifterMask)
  unwrapper = unwrap.MultiFreqPhaseUnwrap(stripeCounts)

  for res in resolutions:
    resPath = expRoot / f"{res[0]}x{res[1]}"
    imgRepo = FileImageRepo(resPath, fileExt='.tif')
    
    cam = camera.FileCamera(camera.CameraConfig(resolution=res[::-1], channels=1),
      imgs=list(imgRepo.GetBy("calibration", sorted=True)))

    minArea = (res[0] * res[1]) / 1000
    maxArea = minArea * 4.5
    calibBoard = board.CircleBoard(circleSpacing=0.03, poiCount=(4, 13), inverted=True, staggered=True, areaHint=(minArea, maxArea))

    calibrator = calib.StereoCalibrator(calibBoard)
    calibrator.Calibrate(cam, proj, shifter, unwrapper, imageCount=boardPoses)

    # Save the experiment information and the calibrated camera / projector
    FileCameraConfigRepo(resPath, overwrite=True).Add(cam.config, "camera")
    FileProjectorRepo(resPath, overwrite=True).Add(proj.config, "projector")

# @pytest.mark.skip(reason="Not ready")
def test_measurement():
  cbSize     = (4, 13)
  squareSize = 0.03
  shifterMask  = 0.03
  phaseCount = 8
  stripeCounts = [1.0, 8.0, 64.0]

  objects = [
    "Hand",
    "Icosphere",
    "Monkey",
    "Donut",
  ]

  # Load projector and camera with imgs
  expRoot = Path(filedialog.askdirectory(title="Where is the folder for the optical devices?"))

  reconstructor = recon.StereoProfil()

  for resolution in resolutions:
    camRepo = FileCameraConfigRepo(expRoot / f"{resolution[0]}x{resolution[1]}", overwrite=True)
    projRepo = FileProjectorRepo(expRoot / f"{resolution[0]}x{resolution[1]}", overwrite=True)

    cam = camera.FileCamera(config=camRepo.Get("camera"))
    proj = utils.FakeFPProjector(config=projRepo.Get("projector"))

    # Check loaded camera resolution is expected
    assert cam.config.resolution == resolution[::-1]

    shifter = shift.NStepPhaseShift(phaseCount, mask=shifterMask)
    unwrapper = unwrap.MultiFreqPhaseUnwrap(stripeCounts)

    # For loading images
    imageRepo = FileImageRepo(expRoot / f"{resolution[0]}x{resolution[1]}", fileExt='.tif', channels=cam.config.channels)

    for obj in objects:
      cam.images = list(imageRepo.GetBy(f"{obj}_", sorted=True))

      pc, _ = reconstructor.Reconstruct(cam, proj, shifter, unwrapper, stripeCounts[-1])

      # Positioned at camera coordinate frame origin (0, 0, 0)
      # offset_x, offset_y = profilometry.checkerboard_centre(cb_size, square_size)
      # pc[:, 0] -= offset_x
      # pc[:, 1] -= offset_y

      # Rotate by 90 degrees to align with OpenGL coordinate frame
      r_x = (np.pi / 2.0) #+ ((15.0 / 180.0) * np.pi)
      R_x = np.array([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)], [0, np.sin(r_x), np.cos(r_x)]])
      # pc = (R_x @ pc.T).T

      save_pointcloud(expRoot / f"{obj}.ply", pc)

      import open3d as o3d
      pcd_load = o3d.io.read_point_cloud(expRoot / f"{obj}.ply")
      o3d.visualization.draw_geometries([pcd_load])

  # TODO: Consider rotation, save as .ply
   
  #pc = pointcloud.rotate_pointcloud(pc, np.pi / 2.0, 0.0, np.pi)

  # From Blender coords to export stl with:
  # Up: Y
  # Forward: -Z