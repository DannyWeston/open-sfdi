import pickle
import cv2
import re
import json

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from .profilometry import IReconstructor
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


# File structure repository

class BaseExperimentRepo(IRepository[IReconstructor]):
    def __init__(self, overwrite):
        self.overwrite = overwrite

    @abstractmethod
    def get(self, id: str) -> IReconstructor:
        pass

    @abstractmethod
    def get_by(self, regex, sorted) -> Iterator[IReconstructor]:
        pass

    @abstractmethod
    def find(self, regex: str, sorted) -> list[str]:
        pass

    @abstractmethod
    def add(self, exp: IReconstructor) -> None:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def update(self, exp: IReconstructor) -> bool:
        pass

class FileExperimentRepo(BaseExperimentRepo):
    data_file = "data.bin"
    manifest_file = "manifest.json"

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

    def get(self, name: str) -> IReconstructor:
        found = self.find(name)

        if len(found) < 1:
            raise Exception(f"Experiment with name '{name}' does not exist")

        return self.__build_exp(name)
    
    def get_by(self, regex, sorted=False) -> Iterator[IReconstructor]:
        # Match all files with correct file extension
        yield from (self.__build_exp(fn) for fn in self.find(regex, sorted))

    def find(self, regex: str, sorted=False) -> list[str]:
        folders = [folder.stem for folder in self.storage_dir.glob("*/")]
        
        folders = list(filter(lambda name: re.match(regex, name), folders))

        if sorted: folders.sort()

        return folders

    def add(self, recon: IReconstructor, name: str) -> None:
        found = self.find(name)

        if 0 < len(found) and (not self.overwrite):
            raise Exception(f"Experiment with name {name} already exists (overwriting disabled)")

        # Make the directory
        folder = (self.storage_dir / name)
        folder.mkdir()

        with open(folder / self.manifest_file, "w") as json_file:
            data = dict()
            data["name"] = name
            data["data"] = self.data_file

            json.dump(data, json_file)

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

    def update(self, exp: IReconstructor) -> bool:
        # TODO: Fix
        self.add(exp)


# Image repositories

class BaseImageRepo(IRepository[Image]):
    @abstractmethod
    def __init__(self, overwrite: bool, greyscale=False):
        self.overwrite = overwrite

        self._greyscale = greyscale

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
    def __init__(self, storage_dir: Path, file_ext='.tif', overwrite=False, greyscale=False):
        super().__init__(overwrite=overwrite, greyscale=greyscale)

        self.storage_dir = storage_dir
        self._file_ext = file_ext

    def __load_img(self, filename):
        return FileImage(self.storage_dir / f"{filename}{self._file_ext}", self._greyscale)

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

class ExperimentService:
    def __init__(self, exp_repo:BaseExperimentRepo, img_repo:BaseImageRepo):
        super().__init__()

        self._exp_repo = exp_repo
        self._img_repo = img_repo

    def save_experiment(self, recon: IReconstructor, name: str):
        try:
            self._exp_repo.add(recon, name)
        except FileExistsError:
            return False
        
        # TODO: Loop through images
        return True

    def save_img(self, img, name) -> bool:
        try:
            self._img_repo.add(img, name)
        except FileExistsError:
            return False
        
        return True

    def load_experiment(self, name) -> IReconstructor:
        return self._exp_repo.get(name)

    def load_img(self, name) -> Image:
        return self._img_repo.get(name)

    def get_by(self, regex, sorted=False) -> Iterator[Image]:
        yield from self._img_repo.get_by(regex, sorted)

    def get_exp_list(self):
        return self._exp_repo.find(".*")

    def exp_exists(self, name):
        return self._exp_repo.find(f"{name}+$") == 1