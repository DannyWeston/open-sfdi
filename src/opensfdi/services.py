import pickle
import cv2
import numpy as np
import re

from PIL import Image
from opensfdi.experiment import Experiment

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

class BaseExperimentRepo(IRepository):
    @abstractmethod
    def get(self, id: str) -> Experiment:
        pass

    @abstractmethod
    def get_all(self) -> list[Experiment]:
        pass

    @abstractmethod
    def find(self, name: str) -> bool:
        pass

    @abstractmethod
    def find_all(self) -> list[str]:
        pass

    @abstractmethod
    def add(self, exp: Experiment) -> None:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def update(self, exp: Experiment) -> bool:
        pass

class FileExperimentRepo(BaseExperimentRepo):
    file_extension = ".opensfdi"
    storage_dir : Path

    def __init__(self, storage_dir : Path):
        super().__init__()

        self.storage_dir = storage_dir

    def get(self, id: str) -> Experiment:
        location = self.storage_dir / f"{id}{self.file_extension}"

        with open(location, "rb") as file:
            profil = pickle.load(file)
            ph_shift = pickle.load(file)
            ph_unwrap = pickle.load(file)

        #return BaseProf.model_validate(raw_bin)

        return Experiment(id, profil, ph_shift, ph_unwrap)
    
    def get_all(self) -> list[Experiment]:
        # Match all files with correct file extension, then use standard getter
        names = [file.stem for file in self.storage_dir.glob(f"*{self.file_extension}")]
        return [self.get(name) for name in names]

    def find(self, name: str) -> bool:
        location = self.storage_dir / f"{name}{self.file_extension}"

        return location.exists()

    def find_all(self) -> list[str]:
        return [file.stem for file in self.storage_dir.glob(f"*{self.file_extension}")]

    def add(self, exp: Experiment):
        location = self.storage_dir / f"{exp.name}{self.file_extension}"

        with open(location, "wb") as file:
            pickle.dump(exp.profil, file)
            pickle.dump(exp.ph_shift, file)
            pickle.dump(exp.ph_unwrap, file)

        # Should now be written !

    def delete(self, name: str) -> bool:
        location = self.storage_dir / f"{name}{self.file_extension}"
        try:
            location.unlink()
            return True

        except FileNotFoundError: 
            return False

    def update(self, exp: Experiment) -> bool:
        # TODO: Fix
        self.add(exp)

# Image repositories

class BaseImageRepo(IRepository):
    @abstractmethod
    def get(self, id: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def add(self, img: np.ndarray, name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def find(self, name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def find_all(self) -> set[str]:
        raise NotImplementedError
    
    # TODO: Implement

    def get_all(self) -> set[np.ndarray]:
        pass

    def delete(self, id) -> bool:
        pass

    def update(self, id, **kwargs) -> bool:
        pass

class FileImageRepo(BaseImageRepo):
    def __init__(self, storage_dir: Path, cache_size=1):
        super().__init__()

        self.storage_dir = storage_dir

        self.cache_size = cache_size # Default to no caching

        self.img_cache = dict()

        self.file_extension = ".png"

    def add(self, img, id: str):
        path = self.storage_dir / f"{id}{self.file_extension}"

        if path.exists(): raise FileExistsError

        # Write image directly if no cache
        if self.cache_size == 1:
            self.__write_img(path, img)
            return

        # Using caching, so check if img is already in cache
        if id in self.img_cache: raise FileExistsError

        self.img_cache[id] = img

        # Write cache if full
        if self.cache_size <= len(self.img_cache):
            for name, img in self.img_cache:
                path = self.storage_dir / f"{name}{self.file_extension}"
                self.__write_img(path, img)
            
            self.img_cache = dict()

    def __write_img(self, path: Path, img):
        pil_img = Image.fromarray(img)
        pil_img.save(str(path.resolve()), quality=100, subsampling=0)

    def get(self, id: str) -> np.ndarray:
        path = self.storage_dir / f"{id}{self.file_extension}"

        # If using cache, check if image is in the cache first before going to disk
        if id in self.img_cache: return self.img_cache[path]

        # Check if img is on disk
        if not path.exists(): raise FileNotFoundError

        # Found on disk so load and return
        return cv2.imread(str(path.resolve()), cv2.IMREAD_UNCHANGED)

    def find(self, name: str) -> bool:
        # Check if in cache
        if name in self.img_cache: return True

        # Check if on disk
        location = self.storage_dir / f"{name}{self.file_extension}"
        return location.exists()

    def find_all(self) -> set[str]:
        # Get cached item ids
        cached = set(self.img_cache.keys())

        # Get items found on disk ids
        on_disk = set([file.stem for file in self.storage_dir.glob(f"*{self.file_extension}")])

        return cached.union(on_disk)


# Services

class ExperimentService:
    def __init__(self, exp_repo:BaseExperimentRepo, img_repo:BaseImageRepo):
        super().__init__()

        self._exp_repo = exp_repo
        self._img_repo = img_repo

    def save_experiment(self, experiment: Experiment):
        self._exp_repo.add(experiment)
        
        # Loop through images

    def save_img(self, img, name):
        self._img_repo.add(img, name)

    def load_experiment(self, name) -> Experiment:
        # TODO: Load images from disk if present

        return self._exp_repo.get(name)

    def load_imgs(self, regex) -> list[np.ndarray]:
        # Filter all available images using regex
        filtered = [name for name in self._img_repo.find_all() if re.match(regex, name)]

        # Load the images using their filtered names
        return [self._img_repo.get(name) for name in filtered]
    
    def get_exp_list(self):
        return self._exp_repo.find_all()
    
    def exp_exists(self, name):
        return self._exp_repo.find(name)