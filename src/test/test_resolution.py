import pytest

from opensfdi import cloud, calibration as calib, reconstruction as recon
from opensfdi.devices import board, camera, projector
from opensfdi.phase import unwrap, shift
from opensfdi.services import FileImageRepo, FileCameraConfigRepo, FileProjectorRepo

from . import utils

resolutions = [
    # (480, 270),
    # (640, 360),
    # (960, 540),
    # (1440, 810),
    (1920, 1080),
    # (2560, 1440),
    # (3840, 2160),
]

@pytest.mark.skip(reason="Not ready")
def test_calibration():
  # Instantiate classes for calibration
  # Load the images to use in the calibration process

  expRoot = utils.DATA_ROOT / "resolution"

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
      images=list(imgRepo.GetBy("calibration", sorted=True)))

    minArea = (res[0] * res[1]) / 1000
    maxArea = minArea * 4.5
    calibBoard = board.CircleBoard(circleSpacing=(0.03, 0.03), poiCount=(4, 7), inverted=True, staggered=True, areaHint=(minArea, maxArea))

    calibrator = calib.StereoCalibrator(calibBoard)
    calibrator.Calibrate(cam, proj, shifter, unwrapper, imageCount=boardPoses)

    # Save the experiment information and the calibrated camera / projector
    FileCameraConfigRepo(resPath, overwrite=True).Add(cam.config, "camera")
    FileProjectorRepo(resPath, overwrite=True).Add(proj.config, "projector")

# @pytest.mark.skip(reason="Not ready")
def test_measurement():
  calibBoard = board.CircleBoard(circleSpacing=(0.03, 0.03), poiCount=(4, 7), inverted=True, staggered=True)

  shifterMask  = 0.03
  phaseCount = 8
  stripeCounts = [1.0, 8.0, 64.0]

  objects = [
    # "Hand",
    # "Icosphere",
    "Monkey",
    # "Donut",
  ]

  # Load projector and camera with imgs
  expRoot = utils.DATA_ROOT / "resolution"

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

      measurementCloud, _ = reconstructor.Reconstruct(cam, proj, shifter, unwrapper, stripeCounts[-1])


      # Positioned at camera coordinate frame origin (0, 0, 0)
      # offset_x, offset_y = profilometry.checkerboard_centre(cb_size, square_size)
      # pc[:, 0] -= offset_x
      # pc[:, 1] -= offset_y

      # Rotate by 90 degrees to align with OpenGL coordinate frame
      
      # Align and save
      measurementCloud = cloud.AlignToCalibBoard(measurementCloud, cam.config, calibBoard)
      cloud.SaveNumpyAsCloud(expRoot / f"{obj}.ply", measurementCloud)

      # Convert to open3d cloud
      measurementCloud = cloud.NumpyToCloud(measurementCloud)

      # Load ground truth cloud
      groundTruthMesh = cloud.LoadMesh(utils.DATA_ROOT / f"{obj}.stl")
      groundTruthCloud = cloud.MeshToCloud(groundTruthMesh)

      cloud.DrawClouds([measurementCloud, groundTruthCloud])

  # TODO: Consider rotation, save as .ply
  #pc = pointcloud.rotate_pointcloud(pc, np.pi / 2.0, 0.0, np.pi)

  # From Blender coords to export stl with:
  # Up: Y
  # Forward: -Z