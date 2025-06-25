import tkinter as tk
import numpy as np
import pytest
import matplotlib.pyplot as plt

from pathlib import Path
from tkinter import filedialog

from opensfdi import profilometry, phase, image, devices
from opensfdi.services import FileImageRepo, FileCameraRepo, FileProjectorRepo, save_pointcloud

# Initialise tkinter for file browsing
# TODO: Change this to use paths etc
root = tk.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()
print("")

# @pytest.mark.skip(reason="Not ready")
# def test_phase():
#   shifts = 12
#   stripe_counts = np.array([1.0, 8.0, 64.0])
#   N = shifts * len(stripe_counts)

#   exp_path = Path(filedialog.askdirectory(title="Where is the folder for the experiment?"))
#   img_repo = FileImageRepo(exp_path / "images", file_ext='.tif', greyscale=True)

#   camera = devices.FileCamera(resolution=(1080, 1920), channels=1,
#     imgs=[img_repo.get("calibration" + str(i+1).zfill(5)) for i in range(N)])

#   projector = devices.FakeProjector(resolution=(1080, 1920))

#   shifter = phase.NStepPhaseShift(shifts)
#   unwrapper = phase.MultiFreqPhaseUnwrap(stripe_counts)
  
#   calibrator = profilometry.StereoCalibrator(cb_size=(10, 7), square_width=0.018, shift_mask=0.1)
#   phasemap = calibrator.gather_phasemap(camera, projector, shifter, unwrapper)

# @pytest.mark.skip(reason="Not ready")
# def test_cv2camera():
#   # camera = devices.CV2Camera(0, resolution=(720, 1280), channels=1)

#   # camera.show_feed()

#   devices_path = Path(filedialog.askdirectory(title="What directory for loading devices?"))
#   cam_repo = FileCameraRepo(devices_path, overwrite=True)

#   camera = cam_repo.get("camera1")

#   camera.show_feed()

# @pytest.mark.skip(reason="Not ready")
# def test_projector():

#   projector = devices.DisplayProjector()

#   devices_path = Path(filedialog.askdirectory(title="What directory for loading devices?"))
#   proj_repo = FileProjectorRepo(devices_path, overwrite=True)

#   proj_repo.add(projector, "projector1")

# @pytest.mark.skip(reason="Not ready")
# def test_gamma():
#   global ex_service
#   imgs = list(ex_service.get_by(f"gamma*", sorted=True))

#   measured_gammas = np.array([image.calc_gamma(img.raw_data) for img in imgs])
#   expected_gammas = np.linspace(0.0, 1.0, len(measured_gammas), dtype=np.float32)

#   image.show_scatter([expected_gammas], [measured_gammas])

# @pytest.mark.skip(reason="Not ready")
# def test_vignette():
#   global ex_service
#   img = ex_service.load_img("gamma1").raw_data

#   diff = (np.ones_like(img, dtype=np.float32) * img.max()) - img
  
#   image.show_image(diff, "Vignetting amount")
#   image.show_image(img + diff, "Vignetting fixed")

# @pytest.mark.skip(reason="Not ready")
# def test_curve():
#   global ex_service
#   imgs = list(ex_service.get_by("intensity", sorted=True))

#   h, w = imgs[0].raw_data.shape

#   d = 25

#   h1 = int(h / 2) - d
#   h2 = h1 + 2 * d

#   w1 = int(w / 2)
#   w2 = w1 + 2 * d

#   intensities = [np.mean(img.raw_data[h1:h2, w1:w2]) for img in imgs]
#   wattages = np.linspace(30.0, 36.625, len(imgs), endpoint=True)

#   plt.ylim(0.0, 1.01)
#   plt.scatter(wattages, intensities, c='r')
#   plt.show()

#   found = next((v for (i, v) in enumerate(wattages[:-1]) if (intensities[i] == intensities[i+1] == 1.0)), None)

#   if not found:
#     print("No max intensity was found given the range of wattages")
#     return True
  
#   print(f"Max intensity wattage: {found}")

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
  exp_root = Path(filedialog.askdirectory(title="Where is the folder for the resolution experiments?"))

  # Fringe projection & calibration parameters
  shift_mask = 0.03
  phase_count = 8
  fringe_counts = [1.0, 8.0, 64.0]
  orientations = 17

  projector = devices.FakeProjector(resolution=(1140, 912), pixel_size=0.8, throw_ratio=1.0)
  shifter = phase.NStepPhaseShift(phase_count, shift_mask=shift_mask)
  unwrapper = phase.MultiFreqPhaseUnwrap(fringe_counts)

  for res in resolutions:
    res_path = exp_root / f"{res[0]}x{res[1]}"
    img_repo = FileImageRepo(res_path, file_ext='.tif')
    
    camera = devices.FileCamera(resolution=res[::-1], channels=1, imgs=list(img_repo.get_by("calibration", sorted=True)))

    area_min = (res[0] * res[1]) / 1000
    area_max = area_min * 4.5
    calib_board = devices.CircleBoard(circle_spacing=0.03, poi_count=(4, 13), inverted=True, staggered=True, area_hint=(area_min, area_max))

    calibrator = profilometry.StereoCalibrator(calib_board)
    calibrator.calibrate(camera, projector, shifter, unwrapper, num_imgs=orientations)

    # Save the experiment information and the calibrated camera / projector
    FileCameraRepo(res_path, overwrite=True).add(camera, "camera")
    FileProjectorRepo(res_path, overwrite=True).add(projector, "projector")

# @pytest.mark.skip(reason="Not ready")
def test_measurement():
  # Fringe projection settings
  cb_size     = (4, 13)
  square_size = 0.03
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