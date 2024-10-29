import numpy as np
import opensfdi.io.repositories as repos

# Image service

class ImageService:
    def __init__(self, img_repo: repos.AbstractImageRepository = repos.FileImageRepository()):
        self._img_repo = img_repo

    def save_image(self, img, name):
        self._img_repo.add(img, name)

    def load_image(self, name, greyscale=False):
        img = self._img_repo.get(name)

        if img is None:
            raise FileNotFoundError(f"Could not find image with name \"{name}\"")

        if img.ndim != 3:
            raise Exception("Image was not loaded in the correct way (incorrect shape)")

        if greyscale:
            # TODO: Allow generic function to convert the image to greyscale
            img = np.mean(img, axis=2)

        return img


# Experiment service

class ExperimentService:
    def __init__(self, ph_calib_repo = repos.FileProfilometryRepo()):
        self._prof_calib_repo = ph_calib_repo

    def save_ph_calib(self, calib, id):
        self._prof_calib_repo.add(calib, id)

    def load_ph_calib(self, name):
        calib = self._prof_calib_repo.get(name)

        if calib is None:
            raise FileNotFoundError(f"Could not find calibration with name \"{name}\"")

        return calib