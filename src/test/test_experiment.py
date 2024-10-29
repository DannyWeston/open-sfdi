import numpy as np
import cv2
import pytest

from pathlib import Path

from opensfdi.io.services import ExperimentService, ImageService
from opensfdi.io.repositories import FileImageRepository, FileProfilometryRepo

from test.video import FakeCamera, FakeFringeProjector

from opensfdi.experiment import FPExperiment
from opensfdi.phase_shifting import NStepPhaseShift
from opensfdi.phase_unwrapping import ReliabilityPhaseUnwrap

from opensfdi.profilometry import LinearInversePH
from opensfdi.utils import show_surface, show_heightmap

meas_path = Path('C:\\Users\\danie\\Desktop\\Results\\Test1\\measurement')
img_service = ImageService(FileImageRepository(meas_path))

test_path = Path('C:\\Users\\danie\\Desktop\\Results\\Test1')
ex_service = ExperimentService(FileProfilometryRepo(test_path))

def test_calibration():
    # Declare paths for loading stuff

    count = 20
    phases = 4
    heights = np.linspace(0.0, 20.0, count, dtype=np.float64)
    imgs = []

    for i in range(count):
        group = []
        for j in range(phases):
            img = img_service.load_image(f"calib_{i}_phase{j}.jpg")
            if img is None: raise Exception(f"Could not load image")
            
            group.append(img)

        imgs.append(np.array(group))

    imgs = np.array(imgs)

    # Get unwrapped phasemaps of loaded images
    print("Starting polynomial calibration...")

    ph_shift = NStepPhaseShift(steps=phases)
    ph_unwrap = ReliabilityPhaseUnwrap()

    z, _, h, w = imgs.shape
    phasemaps = np.empty(shape=(z, h, w))

    for i, height_imgs in enumerate(imgs):
        phasemaps[i] = ph_unwrap.unwrap(ph_shift.shift(height_imgs))


    # Perform calibration
    calib = LinearInversePH()
    calib.calibrate(phasemaps, heights)

    # Load the calibration data
    ex_service.save_ph_calib(calib)

    print(f"Finished calibration")

@pytest.mark.skip(reason="Not ready")
def test_experiment():
    # Load the calibration data
    calib = ex_service.load_ph_calib("linear_inverse0")

    # Load the measurement images
    phases = 12
    imgs = [img_service.load_image(f"measurement_{i}.jpg", greyscale=True) for i in range(phases * 2)]
    print("Finished loading measurement images")

    # Calculate phasemap difference for measurement + reference images
    camera = FakeCamera(imgs)
    projector = FakeFringeProjector()

    ph_shift = NStepPhaseShift(phases)
    ph_unwrap = ReliabilityPhaseUnwrap()

    heightmap = FPExperiment(camera, projector, ph_shift, ph_unwrap, calib).run()

    heightmap[heightmap > 240] = 240
    heightmap[heightmap < 0] = 0

    # Show the result
    show_heightmap(heightmap)