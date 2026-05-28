import cv2
import json_numpy
import os

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from . import utils
from .image import FileImage, Image, ToInt

# Repositories

T = TypeVar('T')
class IRepository(ABC, Generic[T]):
    @abstractmethod
    def Get(self, id) -> T:
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
    def __init__(self, overwrite=True):
        self._overwrite = overwrite
        
    def Get(self, id: str) -> T:
        path = Path(id)

        if not path.exists():
            raise Exception(f"'{path}' could not be found on disk")
        
        with open(path, "r") as jsonFile:
            rawJson = json_numpy.load(jsonFile)

        vc = utils.SerialisableMixin.from_dict(rawJson)

        if vc: return vc
        
        raise Exception(f"Could not construct {T.__class__.__name__} config")

    def Add(self, serialisable: T, id: str) -> None:
        path = Path(id)

        if path.exists() and (not self._overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(path, "w") as jsonFile:
            json_numpy.dump(serialisable.to_dict(), jsonFile, indent=2)

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, serialisable: T) -> bool:
        # TODO: Implement
        raise NotImplementedError

# Basic repository
class FileImageRepo(IRepository[Image]):
    SUPPORTED_FILE_TYPES = [
        ".tif",
        ".tiff",
        ".bmp",
        ".jpg",
        ".jpeg",
        ".png"
    ]

    def __init__(self, overwrite=True):
        self.m_Overwrite = overwrite

    def Add(self, img: Image, id: str):
        ''' Save an image to a repository '''

        _, ext = os.path.splitext(id)

        if ext not in self.SUPPORTED_FILE_TYPES:
            raise Exception(f"Using a file type of '{ext}' is not supported")

        path = Path(id)

        if path.exists() and (not self.m_Overwrite):
            raise FileExistsError(f"Image at {path} already exists (overwriting disabled)")
        
        # TODO: Check file type

        # Save as float to disk
        cv2.imwrite(str(path.resolve()), ToInt(img.raw_data))

    def Get(self, id: str) -> FileImage:
        path = Path(id)

        if not path.exists():
            raise FileNotFoundError(f"Could not find image with id '{id}'")

        return FileImage(Path(path))

    def Delete(self, id) -> bool:
        path = Path(id)

        if not path.exists():
            raise FileNotFoundError(f"Could not find image with id '{id}'")
        
        path.unlink()

    def Update(self, id, **kwargs) -> bool:
        raise NotImplementedError