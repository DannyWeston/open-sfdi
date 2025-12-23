import pickle
import cv2
import re
import json
import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from . import reconstruction as recon

from .devices import camera, projector, characterisation

from .image import FileImage, Image, ToInt


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

class VisionConfigRepo(IRepository[characterisation.Characterisation]):
    @abstractmethod
    def __init__(self, overwrite=True):
        self.m_Overwrite = overwrite

    @abstractmethod
    def Get(self, id: str) -> characterisation.Characterisation:
        pass

    @abstractmethod
    def GetBy(self, regex, sorted) -> Iterator[characterisation.Characterisation]:
        pass

    @abstractmethod
    def Find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def Add(self, config: characterisation.Characterisation, id: str) -> None:
        pass

    @abstractmethod
    def Delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def Update(self, config: characterisation.Characterisation) -> bool:
        pass

class FileVisionConfigRepo(VisionConfigRepo):
    def __init__(self, storageDir: Path, overwrite=True):
        super().__init__(overwrite=overwrite)

        self.m_StorageDir = storageDir

    def Get(self, id: str) -> characterisation.Characterisation:
        found = self.Find(id, sorted=False)

        if len(found) < 1:
            raise Exception(f"Vision config with name '{id}' could not be found on disk")

        vc = self.__LoadConfig(id)

        if vc: return vc
        
        raise Exception("Could not construct vision config")

    def GetBy(self, regex, sorted) -> Iterator[characterisation.Characterisation]:
        yield from (self.__LoadConfig(file) for file in self.Find(regex, sorted))

    def Find(self, regex: str, sorted) -> list[str]:
        files = [file.stem for file in self.m_StorageDir.glob("*.json")]

        files = list(filter(lambda name: re.match(regex, name), files))

        if sorted: files.sort()

        return files

    def Add(self, config: characterisation.Characterisation, id: str) -> None:
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
                "PosePOICoords"     : config.poiCoords.tolist(),
                "BoardPoses"        : config.boardPoses.tolist()
            }

            json.dump(data, jsonFile, indent=2)

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, config: characterisation.Characterisation) -> bool:
        # TODO: Implement
        pass

    def __LoadConfig(self, name):
        with open(self.m_StorageDir / f"{name}.json", "r") as jsonFile:
            rawJson = json.load(jsonFile)

        # Camera is characterised so make calibrated config
        return characterisation.Characterisation(
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

class BaseCameraRepo(IRepository[camera.Camera]):
    @abstractmethod
    def __init__(self, overwrite=True):
        self.m_Overwrite = overwrite

    @abstractmethod
    def Get(self, id: str) -> characterisation.Characterisation:
        pass

    @abstractmethod
    def GetBy(self, regex, sorted) -> Iterator[characterisation.Characterisation]:
        pass

    @abstractmethod
    def Find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def Add(self, camera: camera.Camera, id: str) -> None:
        pass

    @abstractmethod
    def Delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def Update(self, camera: camera.Camera) -> bool:
        pass

class FileCameraRepo(BaseCameraRepo):
    def __init__(self, storageDir: Path, overwrite=True):
        super().__init__(overwrite=overwrite)

        self.m_StorageDir = storageDir

    def Get(self, id: str) -> camera.Camera:
        found = self.Find(id, sorted=False)

        if len(found) < 1:
            raise Exception(f"Camera with name '{id}' could not be found on disk")

        camera = self.__LoadCamera(id)

        if camera: return camera
        
        raise Exception("Could not construct camera")

    def GetBy(self, regex, sorted) -> Iterator[camera.Camera]:
        yield from (self.__LoadCamera(file) for file in self.Find(regex, sorted))

    def Find(self, regex: str, sorted) -> list[str]:
        files = [file.stem for file in self.m_StorageDir.glob("*.json")]

        files = list(filter(lambda name: re.match(regex, name), files))

        if sorted: files.sort()

        return files

    def Add(self, camera: camera.Camera, id: str) -> None:
        found = self.Find(id, False)

        if 0 < len(found) and (not self.m_Overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(self.m_StorageDir / f"{id}.json", "w") as jsonFile:
            data = {
                "Resolution"    : list(camera.resolution),
                "Channels"      : camera.channels,
                "RefreshRate"   : camera.refreshRate,
            }

            json.dump(data, jsonFile, indent=2)

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, camera: camera.Camera) -> bool:
        # TODO: Implement
        pass

    def __LoadCamera(self, name):
        with open(self.m_StorageDir / f"{name}.json", "r") as jsonFile:
            rawJson = json.load(jsonFile)

        return camera.Camera(resolution=tuple(rawJson["Resolution"]), 
                             channels=rawJson["Channels"], 
                             refreshRate=rawJson["RefreshRate"],
                             character=None
        )

# Projector Repositories

class BaseProjectorRepo(IRepository[projector.Projector]):
    @abstractmethod
    def __init__(self, overwrite=True):
        self.m_Overwrite = overwrite

    @abstractmethod
    def Get(self, id: str) -> projector.Projector:
        pass

    @abstractmethod
    def GetBy(self, regex, sorted) -> Iterator[projector.Projector]:
        pass

    @abstractmethod
    def Find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def Add(self, projector: projector.Projector, id: str) -> None:
        pass

    @abstractmethod
    def Delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def Update(self, projector: projector.Projector) -> bool:
        pass

class FileProjectorRepo(BaseProjectorRepo):
    def __init__(self, storageDir: Path, overwrite=True):
        super().__init__(overwrite=overwrite)

        self.m_StorageDir = storageDir

    def Get(self, id: str) -> projector.Projector:
        found = self.Find(id, sorted=False)

        if len(found) < 1:
            raise Exception(f"Projector with name '{id}' could not be found on disk")

        projector = self.__LoadProjector(id)

        if projector: return projector
        
        raise Exception("Could not construct projector")

    def GetBy(self, regex, sorted) -> Iterator[projector.Projector]:
        yield from (self.__LoadProjector(file) for file in self.Find(regex, sorted))

    def Find(self, regex: str, sorted) -> list[str]:
        files = [file.stem for file in self.m_StorageDir.glob("*.json")]

        files = list(filter(lambda name: re.match(regex, name), files))

        if sorted: files.sort()

        return files

    def Add(self, projector: projector.Projector, id: str) -> None:
        found = self.Find(id, False)

        if 0 < len(found) and (not self.m_Overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(self.m_StorageDir / f"{id}.json", "w") as json_file:
            data = {
                "Resolution"    : list(projector.resolution),
                "Channels"      : projector.channels,
                "ThrowRatio"    : projector.throwRatio,
                "AspectRatio"   : projector.aspectRatio,
                "Channels"      : projector.refreshRate
            }

            json.dump(data, json_file, indent=2)

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, projector: projector.FringeProjector) -> bool:
        # TODO: Implement
        pass

    def __LoadProjector(self, name):
        with open(self.m_StorageDir / f"{name}.json", "r") as jsonFile:
            rawJson = json.load(jsonFile)

            return projector.Projector(
                resolution=tuple(rawJson["Resolution"]), 
                channels=rawJson["Channels"],
                refreshRate=rawJson["RefreshRate"],
                throwRatio=rawJson["ThrowRatio"], 
                aspectRatio=rawJson["AspectRatio"],
                character=None
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

        # Save as float to disk
        cv2.imwrite(str(path.resolve()), cv2.cvtColor(ToInt(img), cv2.COLOR_RGB2BGR))

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