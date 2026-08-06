import cv2
import json_numpy
import os
import tempfile

import ffmpeg

from abc import ABC, abstractmethod
from pathlib import Path

from vedo import Points, load as vedo_load

from . import utils
from .image import FileImage, Image, ToInt

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
    def __init__(self, overwrite=True):
        self._overwrite = overwrite

    @property
    def overwrite(self):
        return self._overwrite
        
    def Get(self, id: str) -> utils.SerialisableMixin:
        raise NotImplementedError

    def Add(self, serialisable: utils.SerialisableMixin, id: str) -> None:
        raise NotImplementedError

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, serialisable: utils.SerialisableMixin) -> bool:
        # TODO: Implement
        raise NotImplementedError

class FileJSONRepository(JSONRepository):
    def __init__(self, base_dir:Path=None, overwrite=True):
        super().__init__(overwrite=overwrite)

        if base_dir:
            base_dir.mkdir(exist_ok=True)

            if not self._is_dir_writable(base_dir):
                raise PermissionError(f"No write permissions to {base_dir.absolute()}")
            
            self._base_dir = base_dir
        else: self._base_dir = Path("")

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def _is_dir_writable(self, path: Path) -> bool:
        if not path.exists():
            path = path.parent
        
        # Check write permission
        return os.access(path, os.W_OK)
        
    def GetIDs(self):
        filenames = list(self.base_dir.glob(f'*.json'))

        ids = []
        for file in filenames:
            try: 
                obj = self.Get(file)
                if isinstance(obj, utils.SerialisableMixin): ids.append(filenames)

            except: continue

        return ids

    def Get(self, id: str) -> utils.SerialisableMixin:
        if self.base_dir: path = Path(self.base_dir / f"{id}.json")
        else: path = Path(f"{id}.json")

        if not path.exists():
            raise Exception(f"{id} could not be found in {self.base_dir.absolute()}")
        
        with open(path, "r") as jsonFile:
            raw_json = json_numpy.load(jsonFile)

        vc = utils.SerialisableMixin.from_dict(raw_json)

        if vc: return vc
        
        raise Exception(f"Could not construct {vc.__class__.__name__}")

    def Add(self, serialisable: utils.SerialisableMixin, id: str) -> None:
        if self.base_dir: path = self.base_dir / f"{id}.json"
        else: path = Path(f"{id}.json")

        if path.exists() and (not self.overwrite):
            raise Exception(f"{id} already exists and cannot be saved (overwriting disabled)")

        # Save metadata
        with open(path, "w") as jsonFile:
            json_numpy.dump(serialisable.to_dict(), jsonFile, indent=2)

    def Delete(self, id: str) -> bool:
        # TODO: Implement
        pass

    def Update(self, serialisable: utils.SerialisableMixin) -> bool:
        # TODO: Implement
        raise NotImplementedError
    

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
    def supported_file_types(self):
        raise NotImplementedError

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
    def __init__(self, base_dir:Path=None, overwrite=True, file_ext:str="png"):
        super().__init__(overwrite=overwrite)
        
        if file_ext not in self.supported_file_types:
            raise Exception(f"File type {file_ext} not supported")
        self._file_ext = file_ext

        if base_dir:
            base_dir.mkdir(exist_ok=True)

            if not self._is_dir_writable(base_dir):
                raise PermissionError(f"No write permissions to {base_dir.absolute()}")
            
            self._base_dir = base_dir
        else: self._base_dir = Path("data/imgs")

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
            "tif",
            "tiff",
            "bmp",
            "jpg",
            "jpeg",
            "png"
        ]

    def Add(self, img: Image, id: str):
        ''' Save an image to a repository '''
        file = f"{id}.{self.file_ext}"
        if self.base_dir: path = Path(self.base_dir / file)
        else: path = Path(file)

        if path.exists() and (not self.overwrite):
            raise FileExistsError(f"Image at {path} already exists (overwriting disabled)")

        # Save as float to disk
        cv2.imwrite(str(path.resolve()), ToInt(img.raw_data))

    def Get(self, id: str) -> FileImage:
        path = Path(self.base_dir / f"{id}.{self.file_ext}")

        if not path.exists():
            raise FileNotFoundError(f"Could not find image '{id}'")

        return FileImage(Path(path))

    def Delete(self, id) -> bool:
        file = f"{id}.{self.file_ext}"
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
    
    def Get(self, id): pass

    def Delete(self, id) -> bool: pass

    def Update(self, id, **kwargs) -> bool: pass

class FileVideoRepo(VideoRepo):
    DEFAULT_EXT = ".mp4"

    def __init__(self, resolution: tuple[int, int], fps, base_dir:Path=None, overwrite=True):
        super().__init__(overwrite=overwrite)

        self._base_dir = base_dir

        self._resolution = resolution
        self._fps = fps

        self._recording = False
        self._writer = None

        self._flushed = False

    def init(self):
        if self._recording: return
        
        self._tempfile = tempfile.NamedTemporaryFile(dir=self._base_dir, 
            prefix='recording_',
            suffix=FileVideoRepo.DEFAULT_EXT, 
            delete=False,
        )
        
        self._tempfile.close()

        self._writer = ffmpeg \
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{self.resolution[0]}x{self.resolution[1]}', r=self.fps) \
            .output(self._tempfile.name, vcodec='libx264', crf=23, preset='medium', pix_fmt='yuv420p') \
            .overwrite_output() \
            .global_args('-loglevel', 'error') \
            .global_args('-y') \
            .run_async(pipe_stdin=True, quiet=True)
        
        self._recording = True

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
        if not self._recording: self.init()

        self._writer.stdin.write(img.raw_data.tobytes())
        self._writer.stdin.flush()

    def Flush(self, id: str):
        if not self._recording: return self

        if self._flushed:
            raise Exception("Video repository was already flushed, please create another!")

        self._flushed = True

        self._writer.stdin.close()

        self._writer.wait()

        output_path = self.base_dir / f"{id}{FileVideoRepo.DEFAULT_EXT}"

        # Rename 
        os.replace(self._tempfile.name, output_path)

    def Copy(self):
        return FileVideoRepo(base_dir=self.base_dir, resolution=self.resolution, fps=self.fps, overwrite=self.overwrite)