import numpy as np
import os
import json

from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from pathlib import Path

from opensfdi.io import IRepository
from opensfdi.definitions import CALIBRATION_DIR
from . import phase, stereo

# Need to register new types in here
# TODO: Better model for registering types
profilometry_types = [
    ("classic",     phase.ClassicPH),
    ("polynomial",  phase.PolynomialPH),
]

class IProfilometry(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def heightmap(self, **kwargs):
        pass    
    
    @abstractmethod
    def calibrate(self, **kwargs):
        pass

    @abstractmethod
    @property
    def phasemaps_needed(self) -> int:
        pass

class AbstractProfilometryRepo(IRepository):
    DIR_PREFIX = "calibration"
    METADATA_FILE = "info.json"
    CALIB_FILE = "calib.npy"

class FileProfilometryRepo(AbstractProfilometryRepo):
    def __init__(self, directory=CALIBRATION_DIR):
        self.__directory = Path(directory)

    def __next_inc_folder(search):
        i = 0

        while os.path.exists(search % i):
            i += 1

        return search % i

    def get(self, id:int) -> IProfilometry:
        # Check if calibration folder exists

        calib_name = f"{AbstractProfilometryRepo.DIR_PREFIX}{id}"
        folder = self.__directory / calib_name

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Could not find calibration with name \"{calib_name}\"")
        
        # Check if metadata exists

        meta_path = folder / AbstractProfilometryRepo.METADATA_FILE

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Could not find calibration metadata file for {calib_name}")
        
        with open(meta_path, "r") as meta_file:
            metadata = json.load(meta_file)

        # Try to identify type

        calib_type_name = metadata["type"]
        found = [t[1] for t in profilometry_types if t[0] == calib_type_name]
        if len(found) == 0:
            raise Exception(f"Profilometry technique with name \"{calib_type_name}\" does not exist") 

        calib_type = found[0]

        # Try to load data using path

        data_path = folder / metadata["data_path"]
        
        if not os.path.exists(data_path, "r"):
            raise FileNotFoundError(f"Could not find calibration data for {calib_name}")
        
        with open(data_path, "r") as data_file:
            calib = np.load(data_file)

        return calib_type(calib)

    def add(self, prof: IProfilometry) -> int:
        # Check if calibration type is registered
        calib_type = type(prof)

        found = [x[0] for x in profilometry_types if x[1] == calib_type]

        if len(found) <= 0:
            raise Exception("Could not match profilometry type to registered type")
        
        metadata = {}
        metadata["type"] = found[0]

        # Check if calibration dir exists
        if not os.path.exists(CALIBRATION_DIR):
            raise FileNotFoundError(f"Could not find calibration repository ({folder})")
        
        # Get new calibration directory
        test_dir = folder / f"{FileProfilometryRepo.DIR_PREFIX}%s"
        calib_name = FileProfilometryRepo.__next_inc_folder(test_dir)
        os.mkdir(calib_name)

        calib_name = f"{AbstractProfilometryRepo.DIR_PREFIX}{id}"
        folder = self.__directory / calib_name
        
        # Check if metadata exists

        meta_path = folder / AbstractProfilometryRepo.METADATA_FILE

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Could not find calibration metadata file for {calib_name}")
        
        with open(meta_path, "r") as meta_file:
            metadata = json.load(meta_file)

        # Try to identify type

        calib_type_name = metadata["type"]
        found = [t for t in profilometry_types if t[0] == calib_type_name]
        if len(found) == 0:
            raise Exception(f"Profilometry technique with name \"{calib_type_name}\" does not exist") 

        calib_type = found[0]

        # Try to load data using path

        data_path = folder / metadata["data_path"]
        
        if not os.path.exists(data_path, "r"):
            raise FileNotFoundError(f"Could not find calibration data for {calib_name}")
        
        with open(data_path, "r") as data_file:
            calib = np.load(data_file)

        return calib_type(calib)
        return 0

    def delete(self, id:int) -> None:
        pass

    def update(self, id:int, **kwargs):
        raise NotImplementedError

def show_surface(data):
    hf = plt.figure()

    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))

    ha.plot_surface(X, Y, data)

    plt.show()

def show_heightmap(heightmap, title='Heightmap'):
    x, y = np.meshgrid(range(heightmap.shape[0]), range(heightmap.shape[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, np.transpose(heightmap))
    plt.title(title)
    plt.show()