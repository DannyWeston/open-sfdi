import numpy as np
import pytest

from pathlib import Path
from tkinter import filedialog

from opensfdi import profilometry, phase, devices
from opensfdi.services import FileImageRepo, FileCameraRepo, FileProjectorRepo, save_pointcloud


# 15 images with POIs correctly identified
# Camera reprojection error: 0.037178428515158665
# Projector reprojection error: 0.0155283979445968
# Total reprojection error: 0.030699418960659638

# 15 images with POIs correctly identified
# Camera reprojection error: 0.03725554173322041
# Projector reprojection error: 0.007342073302101905
# Total reprojection error: 0.029090705822883197

# 16 images with POIs correctly identified
# Camera reprojection error: 1.1312295555790386
# Projector reprojection error: 0.005640344633771375
# Total reprojection error: 0.7969578991920931

# 8 images with POIs correctly identified
# Camera reprojection error: 0.13942034175798854
# Projector reprojection error: 0.005610398435355003
# Total reprojection error: 0.09880458961691761

# 14 images with POIs correctly identified
# Camera reprojection error: 0.1625135942431531
# Projector reprojection error: 0.0036810645416931248
# Total reprojection error: 0.11525843539664699

# 11 images with POIs correctly identified
# Camera reprojection error: 0.15348383894417003
# Projector reprojection error: 0.003630297575833046
# Total reprojection error: 0.1087160170904441

# 14 images with POIs correctly identified
# Camera reprojection error: 0.3312366513995426
# Projector reprojection error: 0.0030156654419847937
# Total reprojection error: 0.23437565876181418


@pytest.mark.skip(reason="Not ready")
def test_calibration():
    # Instantiate classes for calibration

    # Fringe projection & calibration parameters
    shift_mask = 0.03
    phase_counts = [6, 7]
    fringe_counts = [1.0, 8.0, 64.0]
    orientations = 17

    exp_root = Path(filedialog.askdirectory(title="Where is the folder for the phases?"))


    camera = devices.FileCamera(resolution=(1080, 1920), channels=1)
    projector = devices.FakeProjector(resolution=(1140, 912), pixel_size=0.8, throw_ratio=1.0)
    calib_board = devices.CircleBoard(circle_spacing=0.03, poi_count=(4, 13), inverted=True, staggered=True, area_hint=(2500, 13000))

    shifter = phase.NStepPhaseShift(shift_mask=shift_mask)
    unwrapper = phase.MultiFreqPhaseUnwrap(fringe_counts)
    calibrator = profilometry.StereoCalibrator(calib_board)

    for phase_count in phase_counts:
        # Set correct phase count
        shifter.phase_count = phase_count

        # Set correct camera images
        phase_path = exp_root / str(phase_count)
        img_repo = FileImageRepo(phase_path, file_ext='.tif')
        camera.imgs = list(img_repo.get_by("calibration", sorted=True))

        calibrator.calibrate(camera, projector, shifter, unwrapper, num_imgs=orientations)

        # Save the experiment information and the calibrated camera / projector
        cam_repo = FileCameraRepo(phase_path, overwrite=True)
        proj_repo = FileProjectorRepo(phase_path, overwrite=True)

        cam_repo.add(camera, "camera")
        proj_repo.add(projector, "projector")

        # TODO: Save experiment results
  
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
  
  cam_repo = FileCameraRepo(calib_path, overwrite=True)
  camera: devices.FileCamera = cam_repo.get("camera")
  img_repo = FileImageRepo(calib_path, file_ext='.tif', channels=camera.channels)

  proj_repo = FileProjectorRepo(calib_path, overwrite=True)
  projector: devices.FakeProjector = proj_repo.get("projector")

  # Phase related stuff
  shifter = phase.NStepPhaseShift(phase_count, shift_mask=shift_mask)
  unwrapper = phase.MultiFreqPhaseUnwrap(num_stripes)
  reconstructor = profilometry.StereoProfil()

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

    save_pointcloud(calib_path / f"{obj}.ply", pc)

    import open3d as o3d
    pcd_load = o3d.io.read_point_cloud(calib_path / f"{obj}.ply")
    o3d.visualization.draw_geometries([pcd_load])

  # TODO: Consider rotation, save as .ply
   
  #pc = pointcloud.rotate_pointcloud(pc, np.pi / 2.0, 0.0, np.pi)

  # From Blender coords to export stl with:
  # Up: Y
  # Forward: -Z