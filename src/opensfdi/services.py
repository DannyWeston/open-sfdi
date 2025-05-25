import pickle
import cv2
import re
import json
import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from . import devices, profilometry
from .image import FileImage, Image, to_f32

# Repositories

T = TypeVar('T')
class IRepository(ABC, Generic[T]):
    @abstractmethod
    def get(self, id) -> T:
        raise NotImplementedError

    @abstractmethod
    def get_by(self, regex, sorted: bool) -> Iterator[T]:
        raise NotImplementedError

    @abstractmethod
    def find(self, regex: str, sorted: bool) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def add(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, id) -> bool:
        raise NotImplementedError

    @abstractmethod
    def update(self, id, **kwargs) -> bool:
        raise NotImplementedError


# Camera repository

class BaseCameraRepo(IRepository[devices.Camera]):
    @abstractmethod
    def __init__(self, overwrite=False):
        self._overwrite = overwrite

    @abstractmethod
    def get(self, id: str) -> devices.Camera:
        pass

    @abstractmethod
    def get_by(self, regex, sorted) -> Iterator[devices.Camera]:
        pass

    @abstractmethod
    def find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def add(self, camera: devices.Camera, id: str) -> None:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def update(self, camera: devices.Camera) -> bool:
        pass

class FileCameraRepo(BaseCameraRepo):
    def __init__(self, storage_dir: Path, overwrite=False):
        super().__init__(overwrite=overwrite)

        self.__storage_dir = storage_dir

    def get(self, id: str) -> devices.Camera:
        found = self.find(id, sorted=False)

        if len(found) < 1:
            raise Exception(f"Camera with name '{id}' could not be found on disk")

        camera = self.__load_camera(id)

        if camera:
            return camera
        
        raise Exception("Could not construct camera")

    def get_by(self, regex, sorted) -> Iterator[devices.Camera]:
        yield from (self.__load_camera(file) for file in self.find(regex, sorted))

    def find(self, regex: str, sorted) -> list[str]:
        files = [file.stem for file in self.__storage_dir.glob("*.json")]

        files = list(filter(lambda name: re.match(regex, name), files))

        if sorted: files.sort()

        return files

    def add(self, camera: devices.Camera, id: str) -> None:
        if not devices.CameraRegistry.is_registered(camera.__class__):
            raise Exception("Camera is not registered to be saved")

        found = self.find(id, False)

        if 0 < len(found) and (not self._overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(self.__storage_dir / f"{id}.json", "w") as json_file:
            data = {
                "camera_type"   : camera.__class__.__name__,
                "is_calibrated"    : camera.is_calibrated(),
                "resolution"    : list(camera.resolution),
                "channels"      : camera.channels
            }

            if camera.is_calibrated():
                data["proj_mat"] = camera.proj_mat.tolist(),
                data["dist_mat"] = camera.dist_mat.tolist(),
                data["intrinsic_mat"] = camera.intrinsic_mat.tolist(),
                data["reproj_error"] = camera.reproj_error,

            # Save any cv2 device stream
            if isinstance(camera, devices.CV2Camera):
                data["cv2_device"] = camera.device

            json.dump(data, json_file, indent=2)

        # Should now be written !

    def delete(self, id: str) -> bool:
        pass

    def update(self, exp: devices.Camera) -> bool:
        pass

    def __load_camera(self, name):
        with open(self.__storage_dir / f"{name}.json", "r") as json_file:
            raw_json = json.load(json_file)

        clazz = devices.CameraRegistry.get_class(raw_json["camera_type"])

        if clazz is None: return None

        camera: devices.Camera = clazz(resolution=tuple(raw_json["resolution"]), channels=1)

        if isinstance(camera, devices.CV2Camera):
            camera.device = raw_json["cv2_device"]

        if raw_json["is_calibrated"]:
            camera.proj_mat = np.array(raw_json["proj_mat"]).reshape((3, 4))
            camera.dist_mat = np.array(raw_json["dist_mat"])
            camera.intrinsic_mat = np.array(raw_json["intrinsic_mat"]).reshape((3, 3))
            camera.reproj_error = raw_json["reproj_error"]

        return camera


# Projector Repositories

class BaseProjectorRepo(IRepository[devices.Projector]):
    @abstractmethod
    def __init__(self, overwrite=False):
        self._overwrite = overwrite

    @abstractmethod
    def get(self, id: str) -> devices.Projector:
        pass

    @abstractmethod
    def get_by(self, regex, sorted) -> Iterator[devices.Projector]:
        pass

    @abstractmethod
    def find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def add(self, projector: devices.Projector, id: str) -> None:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def update(self, projector: devices.Projector) -> bool:
        pass

class FileProjectorRepo(BaseProjectorRepo):
    def __init__(self, storage_dir: Path, overwrite=False):
        super().__init__(overwrite=overwrite)

        self.__storage_dir = storage_dir

    def get(self, id: str) -> devices.Projector:
        found = self.find(id, sorted=False)

        if len(found) < 1:
            raise Exception(f"Projector with name '{id}' could not be found on disk")

        projector = self.__load_projector(id)

        if projector: return projector
        
        raise Exception("Could not construct projector")

    def get_by(self, regex, sorted) -> Iterator[devices.Projector]:
        yield from (self.__load_projector(file) for file in self.find(regex, sorted))

    def find(self, regex: str, sorted) -> list[str]:
        files = [file.stem for file in self.__storage_dir.glob("*.json")]

        files = list(filter(lambda name: re.match(regex, name), files))

        if sorted: files.sort()

        return files

    def add(self, projector: devices.Projector, id: str) -> None:
        if not devices.ProjectorRegistry.is_registered(projector.__class__):
            raise Exception("Projector is not registered to be saved")
        
        found = self.find(id, False)

        if 0 < len(found) and (not self._overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(self.__storage_dir / f"{id}.json", "w") as json_file:
            data = {
                "projector_type"   : projector.__class__.__name__,
                "is_calibrated"    : projector.is_calibrated(),
                "resolution"    : list(projector.resolution),
                "channels"      : projector.channels
            }

            if projector.is_calibrated():
                data["proj_mat"] = projector.proj_mat.tolist(),
                data["dist_mat"] = projector.dist_mat.tolist(),
                data["intrinsic_mat"] = projector.intrinsic_mat.tolist(),
                data["reproj_error"] = projector.reproj_error,

            # Save any cv2 device stream
            if isinstance(projector, devices.DisplayProjector):
                data["display_device"] = True

            json.dump(data, json_file, indent=2)

        # Should now be written !

    def delete(self, id: str) -> bool:
        pass

    def update(self, projector: devices.Projector) -> bool:
        pass

    def __load_projector(self, name):
        with open(self.__storage_dir / f"{name}.json", "r") as json_file:
            raw_json = json.load(json_file)

        clazz = devices.ProjectorRegistry.get_class(raw_json["projector_type"])

        if not clazz: return None

        projector: devices.Projector = clazz(resolution=tuple(raw_json["resolution"]), channels=1)

        if raw_json["is_calibrated"]:
            projector.proj_mat = np.array(raw_json["proj_mat"]).reshape((3, 4))
            projector.dist_mat = np.array(raw_json["dist_mat"])
            projector.intrinsic_mat = np.array(raw_json["intrinsic_mat"]).reshape((3, 3))

            projector.reproj_error = raw_json["reproj_error"]

        return projector


# File structure repository

class BaseExperimentRepo(IRepository[profilometry.IReconstructor]):
    def __init__(self, overwrite):
        self.overwrite = overwrite

    @abstractmethod
    def get(self, id: str) -> profilometry.IReconstructor:
        pass

    @abstractmethod
    def get_by(self, regex, sorted) -> Iterator[profilometry.IReconstructor]:
        pass

    @abstractmethod
    def find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def add(self, exp: profilometry.IReconstructor) -> None:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def update(self, exp: profilometry.IReconstructor) -> bool:
        pass

class FileExperimentRepo(BaseExperimentRepo):
    data_file = "data.bin"
    manifest_file = "metadata.json"

    def __init__(self, storage_dir: Path, overwrite=False):
        super().__init__(overwrite)

        self.storage_dir = storage_dir

    def __build_exp(self, name):
        # Make the directory
        folder = self.storage_dir / name

        with open(folder / self.manifest_file, "r") as json_file:
            raw_json = json.load(json_file)
            
            data_file = raw_json["data"]

            with open(folder / data_file, "rb") as file:
                recon = pickle.load(file)

        return recon

    def get(self, name: str) -> profilometry.IReconstructor:
        found = self.find(name)

        if len(found) < 1:
            raise Exception(f"Experiment with name '{name}' does not exist")

        return self.__build_exp(name)
    
    def get_by(self, regex, sorted=False) -> Iterator[profilometry.IReconstructor]:
        # Match all files with correct file extension
        yield from (self.__build_exp(fn) for fn in self.find(regex, sorted))

    def find(self, regex: str, sorted=False) -> list[str]:
        folders = [folder.stem for folder in self.storage_dir.glob("*/")]
        
        folders = list(filter(lambda name: re.match(regex, name), folders))

        if sorted: folders.sort()

        return folders

    def add(self, recon: profilometry.IReconstructor, name: str) -> None:
        found = self.find(name)

        if 0 < len(found) and (not self.overwrite):
            raise Exception(f"Experiment with name {name} already exists (overwriting disabled)")

        # Make the directory
        folder = (self.storage_dir / name)
        folder.mkdir()

        # Save metadata
        with open(folder / self.manifest_file, "w") as json_file:
            recon.metadata["name"] = name
            recon.metadata["data"] = self.data_file

            json.dump(recon.metadata, json_file, indent=2)

        # Save calibration data
        with open(folder / self.data_file, "wb") as file:
            pickle.dump(recon, file)

        # Should now be written !

    def delete(self, name: str) -> bool:
        found = self.find(name)

        if 0 < len(found):
            path:Path = self.storage_dir / found[0]
            path.unlink()
            return True

        return False

    def update(self, exp: profilometry.IReconstructor) -> bool:
        # TODO: Fix
        self.add(exp)


# Image repositories

class BaseImageRepo(IRepository[Image]):
    @abstractmethod
    def __init__(self, overwrite: bool, greyscale=True):
        self.overwrite = overwrite

        self.__greyscale = greyscale

    @property
    def greyscale(self):
        return self.__greyscale
    
    @greyscale.setter
    def greyscale(self, value):
        self.__greyscale = value

    @abstractmethod
    def get(self, id: str) -> Image:
        raise NotImplementedError
    
    @abstractmethod
    def get_by(self, regex, sorted) -> Iterator[Image]:
        pass

    @abstractmethod
    def add(self, img: Image, name: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def find(self, regex: str, sorted) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, id) -> bool:
        pass

    @abstractmethod
    def update(self, id, **kwargs) -> bool:
        pass

class FileImageRepo(BaseImageRepo):
    def __init__(self, storage_dir: Path, file_ext='.tif', overwrite=False, greyscale=True):
        super().__init__(overwrite=overwrite, greyscale=greyscale)

        self.storage_dir = storage_dir
        self._file_ext = file_ext

    def __load_img(self, filename):
        return FileImage(self.storage_dir / f"{filename}{self._file_ext}", greyscale=self.greyscale)

    def add(self, img: Image, id: str):
        ''' Save an image to a repository '''

        found = self.find(id)

        if 0 < len(found) and (not self.overwrite):
            raise FileExistsError(f"Image with id {found[0]} already exists")

        path:Path = self.storage_dir / found[0]

        # Save as float32 to disk
        cv2.imwrite(str(path.resolve()), cv2.cvtColor(to_f32(img), cv2.COLOR_RGB2BGR))

    def get(self, id: str) -> FileImage:
        found = self.find(id)

        if len(found) < 1:
            raise FileNotFoundError(f"Could not find image with id '{id}'")

        return self.__load_img(found[0])

    def get_by(self, regex, sorted=False) -> Iterator[Image]:
        yield from (self.__load_img(fn) for fn in self.find(regex, sorted))

    def find(self, regex: str, sorted=False) -> list[str]:
        filenames = [file.stem for file in self.storage_dir.glob(f"*{self._file_ext}")]

        filenames = list(filter(lambda filename: re.match(regex, filename), filenames))

        if sorted: filenames.sort()

        return filenames

    # NOT IMPLEMENTED
    def delete(self, id) -> bool:
        raise NotImplementedError

    def update(self, id, **kwargs) -> bool:
        raise NotImplementedError


# Services

class Experiment:
    def __init__(self, name: str, reconst, metadata=None):
        self.name = name
        self.reconst = reconst
        self.metadata = metadata

class ExperimentService:
    def __init__(self, exp_repo:BaseExperimentRepo, img_repo:BaseImageRepo):
        super().__init__()

        self._exp_repo = exp_repo
        self._img_repo = img_repo

    def save_experiment(self, reconst, name):
        try:
            self._exp_repo.add(reconst, name)
        except FileExistsError:
            return False
        
        return True

    def save_img(self, img, name) -> bool:
        try:
            self._img_repo.add(img, name)
        except FileExistsError:
            return False
        
        return True

    def load_experiment(self, name) -> profilometry.IReconstructor:
        return self._exp_repo.get(name)

    def load_img(self, name) -> Image:
        return self._img_repo.get(name)

    def get_by(self, regex, sorted=False) -> Iterator[Image]:
        yield from self._img_repo.get_by(regex, sorted)

    def get_exp_list(self):
        return self._exp_repo.find(".*")

    def exp_exists(self, name):
        return self._exp_repo.find(f"{name}+$") == 1