import cv2
import re
import json

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from . import utils
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

class JSONRepository(IRepository[T]):
    def __init__(self, storage_dir, overwrite=True):
        self._storage_dir = storage_dir
        self._overwrite = overwrite
        
    def Get(self, id: str) -> T:
        found = self.Find(id, sorted=False)

        if len(found) < 1:
            raise Exception(f"'{id}' could not be found on disk")

        vc = self.__LoadConfig(id)

        if vc: return vc
        
        raise Exception(f"Could not construct {T.__class__.__name__} config")

    def GetBy(self, regex, sorted) -> Iterator[T]:
        yield from (self.__LoadConfig(file) for file in self.Find(regex, sorted))

    def Find(self, regex: str, sorted) -> list[str]:
        files = [file.stem for file in self._storage_dir.glob("*.json")]

        files = list(filter(lambda name: re.match(regex, name), files))

        if sorted: files.sort()

        return files

    def Add(self, serialisable: T, id: str) -> None:
        found = self.Find(id, False)

        if 0 < len(found) and (not self._overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(self._storage_dir / f"{id}.json", "w") as jsonFile:
            json.dump(serialisable.to_dict(), jsonFile, indent=2)

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, serialisable: T) -> bool:
        # TODO: Implement
        pass

    def __LoadConfig(self, name):
        with open(self._storage_dir / f"{name}.json", "r") as jsonFile:
            rawJson = json.load(jsonFile)

        return utils.SerialisableMixin.from_dict(rawJson)

# Basic repository

class FileImageRepo(IRepository[Image]):
    SUPPORTED_FILE_TYPES = [
        "tif",
        "bmp"
    ]

    def __init__(self, storageDir: Path, use_ext='tif', overwrite=True):
        self.m_Overwrite = overwrite

        if use_ext not in self.SUPPORTED_FILE_TYPES:
            raise Exception(f"Using a file type of '{use_ext}' is not supported")

        self.m_StorageDir = storageDir
        self.m_FileExt = use_ext

    def __LoadImage(self, filename):
        return FileImage(self.m_StorageDir / f"{filename}.{self.m_FileExt}")

    def Add(self, img: Image, id: str):
        ''' Save an image to a repository '''

        found = self.Find(id)

        if 0 < len(found) and (not self.m_Overwrite):
            raise FileExistsError(f"Image with id {found[0]} already exists")

        path = self.m_StorageDir / f"{id}.{self.m_FileExt}"

        # Save as float to disk
        cv2.imwrite(str(path.resolve()), cv2.cvtColor(ToInt(img), cv2.COLOR_RGB2BGR))

    def Get(self, id: str) -> FileImage:
        found = self.Find(id)

        if len(found) < 1:
            raise FileNotFoundError(f"Could not find image with id '{id}'")

        return self.__LoadImage(id)

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