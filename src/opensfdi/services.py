import pickle
import cv2
import re
import json
import numpy as np
import struct

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from .devices import camera, projector, vision

from . import reconstruction as recon
from .image import FileImage, Image, ToF32

# Misc

def save_pointcloud(filename: Path, arr: np.ndarray):
  with open(filename, "wb") as file:
    file.write(bytes('ply\n', 'utf-8'))
    file.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    file.write(bytes(f'element vertex {arr.shape[0]}\n', 'utf-8'))
    file.write(bytes(f'property float x\n', 'utf-8'))
    file.write(bytes(f'property float y\n', 'utf-8'))
    file.write(bytes(f'property float z\n', 'utf-8'))
    file.write(bytes(f'end_header\n', 'utf-8'))

    for i in range(arr.shape[0]):
      file.write(bytearray(struct.pack("fff", arr[i, 0], arr[i, 1], arr[i, 2])))


# Repositories

T = TypeVar('T')
class IRepository(ABC, Generic[T]):
    @abstractmethod
    def Get(self, id) -> T:
        raise NotImplementedError

    @abstractmethod
    def GetBy(self, regex, sorted: bool) -> Iterator[T]:
        raise NotImplementedError

    @abstractmethod
    def Find(self, regex: str, sorted: bool) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def Add(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def Delete(self, id) -> bool:
        raise NotImplementedError

    @abstractmethod
    def Update(self, id, **kwargs) -> bool:
        raise NotImplementedError


# Camera repository

class BaseCameraConfigRepo(IRepository[camera.CameraConfig]):
    @abstractmethod
    def __init__(self, overwrite=False):
        self.m_Overwrite = overwrite

    @abstractmethod
    def Get(self, id: str) -> camera.CameraConfig:
        pass

    @abstractmethod
    def GetBy(self, regex, sorted) -> Iterator[camera.CameraConfig]:
        pass

    @abstractmethod
    def Find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def Add(self, config: camera.CameraConfig, id: str) -> None:
        pass

    @abstractmethod
    def Delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def Update(self, config: camera.CameraConfig) -> bool:
        pass

class FileCameraConfigRepo(BaseCameraConfigRepo):
    def __init__(self, storage_dir: Path, overwrite=False):
        super().__init__(overwrite=overwrite)

        self.m_StorageDir = storage_dir

    def Get(self, id: str) -> camera.CameraConfig:
        found = self.Find(id, sorted=False)

        if len(found) < 1:
            raise Exception(f"Camera config with name '{id}' could not be found on disk")

        camera = self.__LoadConfig(id)

        if camera: return camera
        
        raise Exception("Could not construct camera")

    def GetBy(self, regex, sorted) -> Iterator[camera.CameraConfig]:
        yield from (self.__LoadConfig(file) for file in self.Find(regex, sorted))

    def Find(self, regex: str, sorted) -> list[str]:
        files = [file.stem for file in self.m_StorageDir.glob("*.json")]

        files = list(filter(lambda name: re.match(regex, name), files))

        if sorted: files.sort()

        return files

    def Add(self, config: camera.CameraConfig, id: str) -> None:
        found = self.Find(id, False)

        if 0 < len(found) and (not self.m_Overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(self.m_StorageDir / f"{id}.json", "w") as jsonFile:
            data = {
                "Resolution"    : list(config.resolution),
                "Channels"      : config.channels
            }

            if isinstance(config, camera.CalibratedCameraConfig):
                vc = config.visionConfig
                data["IntrinsicMat"]        = vc.intrinsicMat.tolist()
                data["Rotation"]            = vc.rotation.tolist()
                data["Translation"]         = vc.translation.tolist()
                data["DistortMat"]          = vc.distortMat.tolist()
                data["ReprojErr"]           = vc.reprojErr
                data["TargetResolution"]    = list(vc.targetResolution)
                data["PosePOICoords"]       = vc.posePOICoords.tolist()

            json.dump(data, jsonFile, indent=2)

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, config: camera.CameraConfig) -> bool:
        # TODO: Implement
        pass

    def __LoadConfig(self, name):
        with open(self.m_StorageDir / f"{name}.json", "r") as jsonFile:
            rawJson = json.load(jsonFile)

        if not rawJson["IntrinsicMat"]:
            return camera.CameraConfig(
                tuple(rawJson["Resolution"]), rawJson["Channels"], 
            )

        # Camera is characterised so make calibrated config
        visionConfig = vision.VisionConfig(
            rotation=np.array(rawJson["Rotation"]),
            translation=np.array(rawJson["Translation"]),
            intrinsicMat=np.array(rawJson["IntrinsicMat"]).reshape((3, 3)),
            distortMat=np.array(rawJson["DistortMat"]),
            reprojErr=rawJson["ReprojErr"],
            targetResolution=rawJson["TargetResolution"],
            posePOICoords=np.array(rawJson["PosePOICoords"])
        )

        return camera.CalibratedCameraConfig(
            tuple(rawJson["Resolution"]), rawJson["Channels"], 
            visionConfig=visionConfig
        )

# Projector Repositories

class BaseProjectorConfigRepo(IRepository[projector.ProjectorConfig]):
    @abstractmethod
    def __init__(self, overwrite=False):
        self.m_Overwrite = overwrite

    @abstractmethod
    def Get(self, id: str) -> projector.ProjectorConfig:
        pass

    @abstractmethod
    def GetBy(self, regex, sorted) -> Iterator[projector.ProjectorConfig]:
        pass

    @abstractmethod
    def Find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def Add(self, config: projector.ProjectorConfig, id: str) -> None:
        pass

    @abstractmethod
    def Delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def Update(self, config: projector.ProjectorConfig) -> bool:
        pass

class FileProjectorRepo(BaseProjectorConfigRepo):
    def __init__(self, storageDir: Path, overwrite=False):
        super().__init__(overwrite=overwrite)

        self.m_StorageDir = storageDir

    def Get(self, id: str) -> projector.ProjectorConfig:
        found = self.Find(id, sorted=False)

        if len(found) < 1:
            raise Exception(f"Projector config with name '{id}' could not be found on disk")

        config = self.__LoadConfig(id)

        if config: return config
        
        raise Exception("Could not construct projector")

    def GetBy(self, regex, sorted) -> Iterator[projector.ProjectorConfig]:
        yield from (self.__LoadConfig(file) for file in self.Find(regex, sorted))

    def Find(self, regex: str, sorted) -> list[str]:
        files = [file.stem for file in self.m_StorageDir.glob("*.json")]

        files = list(filter(lambda name: re.match(regex, name), files))

        if sorted: files.sort()

        return files

    def Add(self, config: projector.ProjectorConfig, id: str) -> None:
        found = self.Find(id, False)

        if 0 < len(found) and (not self.m_Overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(self.m_StorageDir / f"{id}.json", "w") as json_file:
            data = {
                "Resolution"    : list(config.resolution),
                "Channels"      : config.channels,
                "ThrowRatio"    : config.throwRatio,
                "PixelSize"     : config.pixelSize
            }

            if isinstance(config, projector.CalibratedProjectorConfig):
                vc = config.visionConfig
                data["Rotation"]            = vc.rotation.tolist()
                data["Translation"]         = vc.translation.tolist()
                data["IntrinsicMat"]        = vc.intrinsicMat.tolist()
                data["DistortMat"]          = vc.distortMat.tolist()
                data["ReprojErr"]           = vc.reprojErr
                data["TargetResolution"]    = list(vc.targetResolution)
                data["PosePOICoords"]       = vc.posePOICoords.tolist()

            json.dump(data, json_file, indent=2)

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, config: projector.FringeProjector) -> bool:
        # TODO: Implement
        pass

    def __LoadConfig(self, name):
        with open(self.m_StorageDir / f"{name}.json", "r") as jsonFile:
            rawJson = json.load(jsonFile)

            if not rawJson["IntrinsicMat"]:
                return projector.ProjectorConfig(
                    tuple(rawJson["Resolution"]), rawJson["Channels"],
                    rawJson["ThrowRatio"], rawJson["PixelSize"]
                )

            # Projector is characterised so make calibrated config
            visionConfig = vision.VisionConfig(
                rotation=np.array(rawJson["Rotation"]),
                translation=np.array(rawJson["Translation"]),
                intrinsicMat=np.array(rawJson["IntrinsicMat"]).reshape((3, 3)),
                distortMat=np.array(rawJson["DistortMat"]),
                reprojErr=rawJson["ReprojErr"],
                targetResolution=rawJson["TargetResolution"],
                posePOICoords=np.array(rawJson["PosePOICoords"])
            )

            return projector.CalibratedProjectorConfig(
                tuple(rawJson["Resolution"]), rawJson["Channels"],
                rawJson["ThrowRatio"], rawJson["PixelSize"],
                visionConfig
            )


# File structure repository

class BaseExperimentRepo(IRepository[recon.IReconstructor]):
    def __init__(self, overwrite):
        self.overwrite = overwrite

    @abstractmethod
    def Get(self, id: str) -> recon.IReconstructor:
        pass

    @abstractmethod
    def GetBy(self, regex, sorted) -> Iterator[recon.IReconstructor]:
        pass

    @abstractmethod
    def Find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def Add(self, exp: recon.IReconstructor) -> None:
        pass

    @abstractmethod
    def Delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def Update(self, exp: recon.IReconstructor) -> bool:
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

    def Get(self, name: str) -> recon.IReconstructor:
        found = self.Find(name)

        if len(found) < 1:
            raise Exception(f"Experiment with name '{name}' does not exist")

        return self.__build_exp(name)
    
    def GetBy(self, regex, sorted=False) -> Iterator[recon.IReconstructor]:
        # Match all files with correct file extension
        yield from (self.__build_exp(fn) for fn in self.Find(regex, sorted))

    def Find(self, regex: str, sorted=False) -> list[str]:
        folders = [folder.stem for folder in self.storage_dir.glob("*/")]
        
        folders = list(filter(lambda name: re.match(regex, name), folders))

        if sorted: folders.sort()

        return folders

    def Add(self, recon: recon.IReconstructor, name: str) -> None:
        found = self.Find(name)

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

    def Delete(self, name: str) -> bool:
        found = self.Find(name)

        if 0 < len(found):
            path:Path = self.storage_dir / found[0]
            path.unlink()
            return True

        return False

    def Update(self, exp: recon.IReconstructor) -> bool:
        # TODO: Fix
        self.Add(exp)


# Image repositories

class BaseImageRepo(IRepository[Image]):
    @abstractmethod
    def __init__(self, overwrite: bool, channels=1):
        self.overwrite = overwrite

        self.__channels = channels

    @property
    def channels(self):
        return self.__channels
    
    @channels.setter
    def channels(self, value):
        self.__channels = value

    @abstractmethod
    def Get(self, id: str) -> Image:
        raise NotImplementedError
    
    @abstractmethod
    def GetBy(self, regex, sorted) -> Iterator[Image]:
        pass

    @abstractmethod
    def Add(self, img: Image, name: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def Find(self, regex: str, sorted) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def Delete(self, id) -> bool:
        pass

    @abstractmethod
    def Update(self, id, **kwargs) -> bool:
        pass

class FileImageRepo(BaseImageRepo):
    def __init__(self, storage_dir: Path, fileExt='.tif', overwrite=False, channels=1):
        super().__init__(overwrite=overwrite, channels=channels)

        self.storage_dir = storage_dir
        self._file_ext = fileExt

    def __load_img(self, filename):
        return FileImage(self.storage_dir / f"{filename}{self._file_ext}", channels=self.channels)

    def Add(self, img: Image, id: str):
        ''' Save an image to a repository '''

        found = self.Find(id)

        if 0 < len(found) and (not self.overwrite):
            raise FileExistsError(f"Image with id {found[0]} already exists")

        path:Path = self.storage_dir / found[0]

        # Save as float32 to disk
        cv2.imwrite(str(path.resolve()), cv2.cvtColor(ToF32(img), cv2.COLOR_RGB2BGR))

    def Get(self, id: str) -> FileImage:
        found = self.Find(id)

        if len(found) < 1:
            raise FileNotFoundError(f"Could not find image with id '{id}'")

        return self.__load_img(found[0])

    def GetBy(self, regex, sorted=False) -> Iterator[Image]:
        yield from (self.__load_img(fn) for fn in self.Find(regex, sorted))

    def Find(self, regex: str, sorted=False) -> list[str]:
        filenames = [file.stem for file in self.storage_dir.glob(f"*{self._file_ext}")]

        filenames = list(filter(lambda filename: re.match(regex, filename), filenames))

        if sorted: filenames.sort()

        return filenames

    # NOT IMPLEMENTED
    def Delete(self, id) -> bool:
        raise NotImplementedError

    def Update(self, id, **kwargs) -> bool:
        raise NotImplementedError


# Services

class Experiment:
    def __init__(self, name: str, reconst: recon.IReconstructor, metadata=None):
        self.name = name
        self.reconst = reconst
        self.metadata = metadata

class ExperimentService:
    def __init__(self, exp_repo : BaseExperimentRepo, img_repo : BaseImageRepo):
        super().__init__()

        self._exp_repo = exp_repo
        self._img_repo = img_repo

    def save_experiment(self, reconst, name):
        try:
            self._exp_repo.Add(reconst, name)
        except FileExistsError:
            return False
        
        return True

    def save_img(self, img, name, experiment) -> bool:
        try:
            self._img_repo.Add(img, name)
        except FileExistsError:
            return False
        
        return True

    def load_experiment(self, name) -> recon.IReconstructor:
        return self._exp_repo.Get(name)

    def load_img(self, name) -> Image:
        return self._img_repo.Get(name)

    def get_by(self, regex, sorted=False) -> Iterator[Image]:
        yield from self._img_repo.GetBy(regex, sorted)

    def get_exp_list(self):
        return self._exp_repo.Find(".*")

    def exp_exists(self, name):
        return self._exp_repo.Find(f"{name}+$") == 1