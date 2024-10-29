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

class AbstractProfilometryRepo(IRepository):
    METADATA_FILE = "info.json"
    CALIB_FILE = "calib.npy"

class FileProfilometryRepo(AbstractProfilometryRepo):
    def __init__(self):
        super().__init__()

    def get(self, id: Path) -> prof.PhaseHeight:
        # Check if calibration with name exists
        if not id.exists(): return None
        
        # Check if metadata exists
        meta_path = id / AbstractProfilometryRepo.METADATA_FILE

        if not meta_path.exists(): return None
        
        with open(meta_path, "r") as meta_file:
            metadata = json.load(meta_file)


        # Try to identify type
        calib_name = metadata["type"]
        calib_type = calib_type_by_name(calib_name)

        if calib_type is None: return None


        # Try to load data using path
        data_path: Path = id / metadata["data_path"]
        
        if not data_path.exists(): return None
        
        with open(data_path, "rb") as data_file:
            calib_data = np.load(data_file)

        # Some function to resolve calib_type
        return calib_type(calib_data)

    def add(self, prof: prof.PhaseHeight, id: Path):
        # Check if calibration type is registered
        calib_type = type(prof)
        calib_name = calib_name_by_type(calib_type)

        if calib_name is None: raise Exception(f"Could not find a registered calibration type for \"{calib_type}\"")

        
        # Get new calibration directory (make one)
        folder = get_incremental_path(id / calib_name)
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
    def __init__(self, overwrite=True):
        self.__overwrite = overwrite

    def get(self, id: Path):
        if not id.exists(): return None

        return cv2.imread(str(id.resolve()), cv2.IMREAD_UNCHANGED)

    def add(self, img, id: Path):
        if (not id.exists()) or self.__overwrite:
            cv2.imwrite(str(id.resolve()), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])