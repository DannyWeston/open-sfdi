import pickle
import cv2
import re
import json
import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from .devices import camera, projector, vision

from . import reconstruction as recon
from .image import FileImage, Image, ToFloat


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


# Vision Repositories

class VisionConfigRepo(IRepository[vision.VisionConfig]):
    @abstractmethod
    def __init__(self, overwrite=True):
        self.m_Overwrite = overwrite

    @abstractmethod
    def Get(self, id: str) -> vision.VisionConfig:
        pass

    @abstractmethod
    def GetBy(self, regex, sorted) -> Iterator[vision.VisionConfig]:
        pass

    @abstractmethod
    def Find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def Add(self, config: vision.VisionConfig, id: str) -> None:
        pass

    @abstractmethod
    def Delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def Update(self, config: vision.VisionConfig) -> bool:
        pass

class FileVisionConfigRepo(VisionConfigRepo):
    def __init__(self, storageDir: Path, overwrite=True):
        super().__init__(overwrite=overwrite)

        self.m_StorageDir = storageDir

    def Get(self, id: str) -> vision.VisionConfig:
        found = self.Find(id, sorted=False)

        if len(found) < 1:
            raise Exception(f"Vision config with name '{id}' could not be found on disk")

        vc = self.__LoadConfig(id)

        if vc: return vc
        
        raise Exception("Could not construct vision config")

    def GetBy(self, regex, sorted) -> Iterator[vision.VisionConfig]:
        yield from (self.__LoadConfig(file) for file in self.Find(regex, sorted))

    def Find(self, regex: str, sorted) -> list[str]:
        files = [file.stem for file in self.m_StorageDir.glob("*.json")]

        files = list(filter(lambda name: re.match(regex, name), files))

        if sorted: files.sort()

        return files

    def Add(self, config: vision.VisionConfig, id: str) -> None:
        found = self.Find(id, False)

        if 0 < len(found) and (not self.m_Overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(self.m_StorageDir / f"{id}.json", "w") as jsonFile:
            data = {
                "IntrinsicMat"      : config.intrinsicMat.tolist(),
                "Rotation"          : config.rotation.tolist(),
                "Translation"       : config.translation.tolist(),
                "DistortMat"        : config.distortMat.tolist(),
                "ReprojErr"         : config.reprojErr,
                "TargetResolution"  : list(config.targetResolution),
                "PosePOICoords"     : config.posePOICoords.tolist(),
                "BoardPoses"        : config.boardPoses.tolist()
            }

            json.dump(data, jsonFile, indent=2)

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, config: vision.VisionConfig) -> bool:
        # TODO: Implement
        pass

    def __LoadConfig(self, name):
        with open(self.m_StorageDir / f"{name}.json", "r") as jsonFile:
            rawJson = json.load(jsonFile)

        # Camera is characterised so make calibrated config
        return vision.VisionConfig(
            rotation =          np.asarray(rawJson["Rotation"]),
            translation =       np.asarray(rawJson["Translation"]),
            intrinsicMat =      np.asarray(rawJson["IntrinsicMat"]).reshape((3, 3)),
            distortMat =        np.asarray(rawJson["DistortMat"]),
            reprojErr =         rawJson["ReprojErr"],
            targetResolution =  rawJson["TargetResolution"],
            posePOICoords =     np.asarray(rawJson["PosePOICoords"]),
            boardPoses =        np.asarray(rawJson["BoardPoses"])
        )


# Camera Repositories

class BaseCameraConfigRepo(IRepository[vision.VisionConfig]):
    @abstractmethod
    def __init__(self, overwrite=True):
        self.m_Overwrite = overwrite

    @abstractmethod
    def Get(self, id: str) -> vision.VisionConfig:
        pass

    @abstractmethod
    def GetBy(self, regex, sorted) -> Iterator[vision.VisionConfig]:
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
    def __init__(self, storage_dir: Path, overwrite=True):
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

        return camera.CameraConfig(
            tuple(rawJson["Resolution"]), rawJson["Channels"], 
        )


# Projector Repositories

class BaseProjectorConfigRepo(IRepository[projector.ProjectorConfig]):
    @abstractmethod
    def __init__(self, overwrite=True):
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
    def __init__(self, storageDir: Path, overwrite=True):
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

            return projector.ProjectorConfig(
                tuple(rawJson["Resolution"]), rawJson["Channels"],
                rawJson["ThrowRatio"], rawJson["PixelSize"]
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
    def __init__(self, overwrite: bool):
        self.overwrite = overwrite

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
    SUPPORTED_FILE_TYPES = [
        "tif",
        "bmp"
    ]

    def __init__(self, storageDir: Path, useExt='tif', overwrite=True):
        super().__init__(overwrite=overwrite)

        if useExt not in self.SUPPORTED_FILE_TYPES:
            raise Exception(f"Using a file type of '{useExt}' is not supported")

        self.m_StorageDir = storageDir
        self.m_FileExt = useExt

    def __LoadImage(self, filename):
        return FileImage(self.m_StorageDir / f"{filename}.{self.m_FileExt}")

    def Add(self, img: Image, id: str):
        ''' Save an image to a repository '''

        found = self.Find(id)

        if 0 < len(found) and (not self.overwrite):
            raise FileExistsError(f"Image with id {found[0]} already exists")

        path = self.m_StorageDir / f"{found[0]}.{self.m_FileExt}"

        # Save as float32 to disk
        cv2.imwrite(str(path.resolve()), cv2.cvtColor(ToFloat(img), cv2.COLOR_RGB2BGR))

    def Get(self, id: str) -> FileImage:
        found = self.Find(id)

        if len(found) < 1:
            raise FileNotFoundError(f"Could not find image with id '{id}'")

        return self.__LoadImage(found[0])

    def GetBy(self, regex, sorted=False) -> Iterator[Image]:
        yield from (self.__LoadImage(fn) for fn in self.Find(regex, sorted))

    def Find(self, regex: str, sorted=False) -> list[str]:
        filenames = [file.stem for file in self.m_StorageDir.glob(f"*.{self.m_FileExt}")]

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