import json
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
    def add(self, **kwargs):
        pass

    @abstractmethod
    def delete(self, id):
        pass

    @abstractmethod
    def update(self, id, **kwargs):
        pass


# Profilometry repositories

class AbstractProfilometryRepo(IRepository):
    DIR_PREFIX = "calibration"
    METADATA_FILE = "info.json"
    CALIB_FILE = "calib.npy"


# File structure repository

# Need to register new types in here
# TODO: Better model for registering types
# Note: If you want to derive your own profilometry techniques, its important to register them in CALIB_TYPES!
CALIB_TYPES = [
    ("classic",     prof.PhaseHeight),
    ("linear_inverse", prof.LinearInversePH),
    ("polynomial",  prof.PolynomialPH),
]

def calib_type_by_name(name):
    found = [x for x in CALIB_TYPES if x[0] == name]

    if len(found) == 0: return None

    return found[0][1]

def calib_name_by_type(calib_type):
    found = [x for x in CALIB_TYPES if x[1] == calib_type]

    if len(found) == 0: return None

    return found[0][0]

def get_incremental_path(search):
    i = 0

    test = None

    while True:
        test = Path(f"{str(search)}{i}")

        if not test.exists(): break

        i += 1

    return test

class FSConfig:
    def __init__(self, ROOT_DIR: Path):
        self.__root_dir = ROOT_DIR                          # Root of the entire codebase
        self.__data_dir = ROOT_DIR / "data"                 # IO data location
        self.__results_dir = self.__data_dir / "results"           # Directory for results to be written to
        self.__fringes_dir = self.__data_dir / "fringes"           # Fringes are to be used from this directory
        self.__calibration_dir = self.__data_dir / "calibration"   # Location where calibration data is dumped

    def make_structure(self):
        self.root_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.calibration_dir.mkdir(exist_ok=True)
        self.fringes_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

    @property
    def root_dir(self) -> Path:
        return self.__root_dir
    
    @property
    def data_dir(self) -> Path:
        return self.__data_dir
    
    @property
    def results_dir(self) -> Path:
        return self.__results_dir
    
    @property
    def fringes_dir(self) -> Path:
        return self.__fringes_dir
    
    @property
    def calibration_dir(self) -> Path:
        return self.__calibration_dir

class FileProfilometryRepo(AbstractProfilometryRepo):
    def __init__(self, output_dir: Path):
        self.__output_dir = output_dir

    def get(self, id) -> prof.PhaseHeight:
        # Check if calibration with name exists
        folder = self.__output_dir / id

        if not folder.exists(): return
        
        # Check if metadata exists
        meta_path = folder / AbstractProfilometryRepo.METADATA_FILE

        if not meta_path.exists(): return None
        
        with open(meta_path, "r") as meta_file:
            metadata = json.load(meta_file)


        # Try to identify type
        calib_name = metadata["type"]
        calib_type = calib_type_by_name(calib_name)

        if calib_type is None: return None


        # Try to load data using path
        data_path: Path = folder / metadata["data_path"]
        
        if not data_path.exists(): return None
        
        with open(data_path, "rb") as data_file:
            calib_data = np.load(data_file)

        # Some function to resolve calib_type
        return calib_type(calib_data)

    def add(self, prof: prof.PhaseHeight):
        # Check if calibration type is registered
        calib_type = type(prof)
        calib_name = calib_name_by_type(calib_type)

        if calib_name is None: raise Exception(f"Could not find a registered calibration type for \"{calib_type}\"")

        
        # Get new calibration directory (make one)
        folder = get_incremental_path(self.__output_dir / calib_name)
        folder.mkdir(exist_ok=True) # Shouldn't exist already, but ignore if it does


        # Make metadata file
        meta_path = folder / AbstractProfilometryRepo.METADATA_FILE
        
        metadata = dict()
        metadata["type"] = calib_name
        metadata["data_path"] = AbstractProfilometryRepo.CALIB_FILE

        with open(meta_path, "w") as meta_file:
            metadata = json.dump(metadata, meta_file, indent=4)


        # Write data to disk
        with open(folder / AbstractProfilometryRepo.CALIB_FILE, "wb") as data_file:
            np.save(data_file, prof.calib_data)

    # Not needed !
    def delete(self, id:int) -> None:
        pass

    def update(self, id:int, **kwargs):
        raise NotImplementedError


# Image repositories

class AbstractImageRepository(IRepository):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get(self, id):
        pass

    @abstractmethod
    def add(self, **kwargs):
        pass

    # Don't need implementations

    def delete(self, id):
        pass

    def update(self, id, **kwargs):
        pass

class FileImageRepository(AbstractImageRepository):
    DEFAULT_EXT = ".png"

    def __init__(self, output_dir: Path):
        self.__output_dir = output_dir

    def get(self, name: str):
        if name is None: raise TypeError
        
        path = self.__output_dir / f"{name}{FileImageRepository.DEFAULT_EXT}"

        if not path.exists(): return None

        return cv2.imread(str(path.resolve()), cv2.IMREAD_UNCHANGED)

    def add(self, img, name):
        path = self.__output_dir / f"{name}{FileImageRepository.DEFAULT_EXT}"
        
        if path.exists(path):
            path = get_incremental_path(path)

        cv2.imwrite(str(path.resolve()), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])