import pickle
import cv2
import numpy as np
import opensfdi.profilometry as prof

from abc import ABC, abstractmethod
from pathlib import Path

# Repositories

class IRepository(ABC):
    @abstractmethod
    def get(self, id):
        pass

    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def find(self, name) -> bool:
        pass

    @abstractmethod
    def find_all(self) -> list[str]:
        pass

    @abstractmethod
    def add(self, **kwargs):
        pass

    @abstractmethod
    def delete(self, id):
        pass

    @abstractmethod
    def update(self, id, **kwargs):
        pass


# File structure repository

class BaseProfRepo(IRepository):
    @abstractmethod
    def get(self, id: str) -> prof.BaseProf:
        pass

    @abstractmethod
    def get_all(self) -> list[prof.BaseProf]:
        pass

    @abstractmethod
    def find(self, name: str) -> bool:
        pass

    @abstractmethod
    def find_all(self) -> list[str]:
        pass

    @abstractmethod
    def add(self, prof: prof.BaseProf) -> None:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def update(self, prof: prof.BaseProf) -> bool:
        pass

class FileProfRepo(BaseProfRepo):
    file_extension = ".opensfdi"
    storage_dir : Path

    def __init__(self, storage_dir : Path):
        super().__init__()
        self.storage_dir = storage_dir

    def get(self, id: str) -> prof.BaseProf:
        with open(self.storage_dir / f"{id}{self.file_extension}", "rb") as in_file:
            raw_bin = pickle.load(in_file)

        return prof.BaseProf.model_validate(raw_bin)
    
    def get_all(self) -> list[prof.BaseProf]:
        # Match all files with correct file extension, then use standard getter
        names = [file.stem for file in self.storage_dir.glob(f"*{self.file_extension}")]
        return [self.get(name) for name in names]

    def find(self, name: str) -> bool:
        return (self.storage_dir / f"{name}{self.file_extension}").exists()

    def find_all(self) -> list[str]:
        return [file.stem for file in self.storage_dir.glob(f"*{self.file_extension}")]

    def add(self, prof: prof.BaseProf):
        raw_bin = pickle.dumps(prof)
        with open(self.storage_dir / f"{prof.name}{self.file_extension}", "wb") as out_file:
            raw_bin = out_file.write(raw_bin)

        # Should now be written !

    def delete(self, name: str) -> bool:
        try:
            file = self.storage_dir / f"{name}{self.file_extension}"
            file.unlink()
            return True

        except FileNotFoundError: 
            return False

    def update(self, prof: prof.BaseProf) -> bool:
        # TODO: Fix
        self.add(prof)


# Image repositories

class BaseImageRepo(IRepository):
    @abstractmethod
    def get(self, id: str) -> np.ndarray:
        pass

    @abstractmethod
    def add(self, img: np.ndarray, name: str) -> None:
        pass

    # Don't need implementations

    def find(self, name: str) -> bool:
        pass

    def find_all(self) -> list[str]:
        pass
    
    def get_all(self) -> list[prof.BaseProf]:
        pass

    def delete(self, id) -> bool:
        pass

    def update(self, id, **kwargs) -> bool:
        pass

class FileImageRepo(BaseImageRepo):
    storage_dir : Path

    def __init__(self, storage_dir : Path):
        super().__init__()
        self.storage_dir = storage_dir

    def add(self, img, id: str):
        path = self.storage_dir / id

        if path.exists(): raise FileExistsError

        cv2.imwrite(str(path.resolve()), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def get(self, id: str) -> np.ndarray:
        path = self.storage_dir / id

        if not path.exists(): raise FileNotFoundError

        return cv2.imread(str(path.resolve()), cv2.IMREAD_UNCHANGED)


# Services

class ExperimentService:
    _prof_repo : BaseProfRepo

    def __init__(self, prof_repo : BaseProfRepo):
        super().__init__()
        self._prof_repo = prof_repo

    @property
    def prof_repo(self) -> BaseProfRepo:
        return self._prof_repo

    @prof_repo.setter
    def prof_repo(self, prof_repo: BaseProfRepo):
        self._prof_repo = prof_repo

    def save_calib(self, experiment):
        self.prof_repo.add(experiment)

    def load_calib(self, name):
        return self.prof_repo.get(name)
    
    def get_calib_list(self):
        return self.prof_repo.find_all()
    
    def calib_exists(self, name):
        return self.prof_repo.find(name)

class ImageService:
    _img_repo : BaseImageRepo

    def __init__(self, img_repo : BaseImageRepo):
        super().__init__()
        self._img_repo = img_repo

    @property
    def img_repo(self) -> BaseImageRepo:
        return self._img_repo

    @img_repo.setter
    def img_repo(self, img_repo: BaseImageRepo):
        self._img_repo = img_repo

    def save_image(self, img, name):
        self.img_repo.add(img, name)

    def load_image(self, name):
        return self.img_repo.get(name)