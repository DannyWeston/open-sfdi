from typing import Generic, TypeVar

import platformdirs

import cv2
import json_numpy
import os
import tempfile

import ffmpeg

from abc import ABC, abstractmethod
from pathlib import Path

from vedo import Points, load as vedo_load

from .. import utils
from ..image import FileImage, Image, ToInt

# Repositories

class IRepository(ABC):
    @abstractmethod
    def Get(self, id):
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

class JSONRepository(IRepository):
    def __init__(self, base_dir:Path, overwrite=True):
        super().__init__()

        self._overwrite = overwrite

        self._base_dir = base_dir

    @property
    def overwrite(self):
        return self._overwrite

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def GetIDs(self):
        filenames = list(self.base_dir.glob(f'*.json'))

        ids = []
        for file in filenames:
            try: 
                obj = self.Get(file)
                if isinstance(obj, utils.SerialisableMixin): ids.append(filenames)

            except: continue

        return ids

    def Get(self, id: str) -> dict:
        path = Path(self.base_dir / f"{id}.json")

        if not path.exists():
            raise Exception(f"{id} could not be found in {self.base_dir.absolute()}")
        
        with open(path, "r") as json_file:
            raw_json = json_numpy.load(json_file)
            return raw_json
        
        raise Exception(f"Could not find json")

    def Add(self, data: dict, id: str) -> None:
        path = Path(self.base_dir / f"{id}.json")

        if path.exists() and (not self.overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(path, "w") as json_file:
            return json_numpy.dump(data, json_file, indent=4)

        raise Exception(f"Could not find json")

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, data: dict) -> bool:
        # TODO: Implement
        pass

# Pointcloud Repository
class PointCloudRepo(IRepository):
    def __init__(self, overwrite=True):
        self._overwrite = overwrite

    @property
    def overwrite(self):
        return self._overwrite

    def Add(self, cloud: Points, id: str):
        raise NotImplementedError

    def Get(self, id: str) -> Points:
        raise NotImplementedError

    def Delete(self, id) -> bool:
        raise NotImplementedError

    def Update(self, id, **kwargs) -> bool:
        raise NotImplementedError
    
class PointCloudFileRepo(PointCloudRepo):
    def __init__(self, base_dir:Path=None, overwrite=True, file_ext:str="ply"):
        super().__init__(overwrite=overwrite)
        
        if file_ext not in self.supported_file_types:
            raise Exception(f"File type {file_ext} not supported")
        self._file_ext = file_ext

        self._base_dir = Path("data/measurements") if base_dir is None else base_dir

        self._base_dir.mkdir(exist_ok=True)

        if not self._is_dir_writable(self._base_dir):
            raise PermissionError(f"No write permissions to {self._base_dir.absolute()}")

    @property
    def base_dir(self) -> Path:
        return self._base_dir
    
    @base_dir.setter
    def base_dir(self, value: Path):
        if value:
            if not value.is_dir():
                raise Exception("Base directory is a file!")
            
            if not self._is_dir_writable(value):
                raise PermissionError(f"No write permissions to {value.absolute()}")

        self._base_dir = value

    @property
    def file_ext(self):
        return self._file_ext
    
    @file_ext.setter
    def file_ext(self, value: str):
        if value not in self.supported_file_types:
            raise Exception(f"File type {value} not supported")

        self._file_ext = value

    def _is_dir_writable(self, path: Path) -> bool:
        path = Path(path)
        
        if not path.exists():
            path = path.parent
        
        # Check write permission
        return os.access(path, os.W_OK)

    @property
    def supported_file_types(self):
        return [
            "ply",
            "obj",
            "stl",
            "vtp",
            "xyz",
            "pcd"
        ]

    def Add(self, cloud: Points, id: str):
        ''' Save an point-cloud to a repository '''
        file = f"{id}.{self.file_ext}"
        if self.base_dir: path = Path(self.base_dir / file)
        else: path = Path(file)

        if path.exists() and (not self.overwrite):
            raise FileExistsError(f"Point-cloud at {path} already exists (overwriting disabled)")

        # Save as float to disk
        cloud.write(str(path.resolve()))

    def Get(self, id: str) -> Points:
        path = Path(self.base_dir / f"{id}.{self.file_ext}")

        if not path.exists():
            raise FileNotFoundError(f"Could not find point-cloud '{id}'")

        return vedo_load(str(path.absolute()))

    def Delete(self, id) -> bool:
        file = f"{id}.{self.file_ext}"
        if self.base_dir: path = Path(self.base_dir / file)
        else: path = Path(file)

        if not path.exists():
            raise FileNotFoundError(f"Could not find point-cloud with id '{id}'")
        
        path.unlink()

    def Update(self, id, **kwargs) -> bool:
        raise NotImplementedError


# Image repository

class ImageRepo(IRepository):
    def __init__(self, overwrite=True):
        self._overwrite = overwrite

    @property
    def overwrite(self):
        return self._overwrite

    def Add(self, img: Image, id: str):
        raise NotImplementedError

    def Get(self, id: str) -> Image:
        raise NotImplementedError

    def Delete(self, id) -> bool:
        raise NotImplementedError

    def Update(self, id, **kwargs) -> bool:
        raise NotImplementedError

class FileImageRepo(ImageRepo):
    DEFAULT_DIR = Path(platformdirs.user_pictures_path())
    SUPPORTED_FILE_EXTS = [
        ".bmp",
        ".tif",
        ".jpg",
        ".jpeg",
        ".png"
    ]

    def __init__(self, base_dir:Path=None, overwrite=True):
        super().__init__(overwrite=overwrite)
        
        self._base_dir = base_dir if base_dir else FileImageRepo.DEFAULT_DIR

    @property
    def base_dir(self) -> Path:
        return self._base_dir
    
    @base_dir.setter
    def base_dir(self, value: Path):
        if value:
            if not value.is_dir():
                raise Exception("Base directory is a file!")
            
            if not self._is_dir_writable(value):
                raise PermissionError(f"No write permissions to {value.absolute()}")

        self._base_dir = value

    def _is_dir_writable(self, path: Path) -> bool:
        path = Path(path)
        
        if not path.exists():
            path = path.parent
        
        # Check write permission
        return os.access(path, os.W_OK)

    def Add(self, img: Image, id: str, file_ext:str="bmp"):
        ''' Save an image to a repository '''
        if file_ext not in self.SUPPORTED_FILE_EXTS:
            raise Exception(f"File type {file_ext} not supported")

        path = self.base_dir / f"{id}{file_ext}"

        if path.exists() and (not self.overwrite):
            raise FileExistsError(f"Image at {path} already exists (overwriting disabled)")

        # Save as float to disk
        cv2.imwrite(str(path.resolve()), ToInt(img.raw_data))

    def Get(self, id: str, file_ext=str) -> FileImage:
        path = Path(self.base_dir / f"{id}{file_ext}")

        if file_ext not in self.SUPPORTED_FILE_EXTS:
            raise Exception(f"File type {file_ext} not supported")

        if not path.exists():
            raise FileNotFoundError(f"Could not find image '{id}'")

        return FileImage(Path(path))

    def Delete(self, id, file_ext=str) -> bool:
        file = f"{id}{file_ext}"
        if self.base_dir: path = Path(self.base_dir / file)
        else: path = Path(file)

        if not path.exists():
            raise FileNotFoundError(f"Could not find image with id '{id}'")
        
        path.unlink()

    def Update(self, id, **kwargs) -> bool:
        raise NotImplementedError

# Video Repository

class VideoRepo(IRepository):
    def __init__(self, overwrite=True):
        super().__init__()

        self._overwrite = overwrite

    @property
    def overwrite(self):
        return self._overwrite

    def Add(self, img: Image): raise NotImplementedError
    
    def Get(self, id): raise NotImplementedError

    def Delete(self, id) -> bool: raise NotImplementedError

    def Update(self, id, **kwargs) -> bool: raise NotImplementedError

    def Flush(self): raise NotImplementedError

class FileVideoRepo(VideoRepo):
    DEFAULT_EXT = ".mp4"
    DEFAULT_DIR = Path(platformdirs.user_videos_dir())
    TEMP_DIR = Path(platformdirs.user_videos_dir())

    SUPPORTED_EXTENSIONS = [
        ".mp4", 
        ".mov", 
        ".avi"
    ]

    def __init__(self, resolution: tuple[int, int], fps, base_dir:Path=None, file_ext:str=None, overwrite=True):
        super().__init__(overwrite=overwrite)

        self._base_dir = base_dir if base_dir is not None else FileVideoRepo.DEFAULT_DIR

        if file_ext not in FileVideoRepo.SUPPORTED_EXTENSIONS:
            raise Exception("File extension not supported!")

        self._file_ext = file_ext

        self._resolution = resolution
        self._fps = fps

        self._flushed = False

        # Make a temporary file to hold the recording
        temp_file = tempfile.NamedTemporaryFile(dir=FileVideoRepo.TEMP_DIR, 
            prefix='recording_',
            suffix=self._file_ext,
            delete=False,
        )

        self._cache_path = FileVideoRepo.TEMP_DIR / temp_file.name

        temp_file.close()

        self._writer = ffmpeg \
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{self.resolution[0]}x{self.resolution[1]}', use_wallclock_as_timestamps=True) \
            .output(str(self._cache_path), vcodec='libx264', crf=23, preset='medium', pix_fmt='yuv420p') \
            .overwrite_output() \
            .global_args('-loglevel', 'error') \
            .global_args('-y') \
            .run_async(pipe_stdin=True)

    def _is_dir_writable(self, path: Path) -> bool:
        path = Path(path)
        
        if not path.exists():
            path = path.parent
        
        # Check write permission
        return os.access(path, os.W_OK)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value: Path):
        if value:
            if not value.is_dir():
                raise Exception("Base directory is a file!")
            
            if not self._is_dir_writable(value):
                raise PermissionError(f"No write permissions to {value.absolute()}")

        self._base_dir = value

    @property
    def resolution(self):
        return self._resolution

    @property
    def fps(self):
        return self._fps

    def Add(self, img: Image):
        if self._flushed: return

        self._writer.stdin.write(img.raw_data.tobytes())
        self._writer.stdin.flush()

    def Flush(self, id: str=None):
        # TODO: Check if any frames were actually written

        if self._flushed:
            raise Exception("Video repository was already flushed, please create another!")

        self._flushed = True

        self._writer.stdin.close()

        self._writer.wait()

        output_path = self.base_dir / f"{id}{self._file_ext}"

        # Rename
        if id is not None:
            os.replace(self._cache_path, output_path)

    def Copy(self):
        return FileVideoRepo(base_dir=self.base_dir, resolution=self.resolution, fps=self.fps, overwrite=self.overwrite)