import cv2
import os
import pickle

from abc import ABC, abstractmethod

class Repo(ABC):
    @abstractmethod
    def commit(self):
        pass

class ImageRepo(Repo):
    @abstractmethod
    def add_image(self, img, name):
        raise NotImplementedError
    
    @abstractmethod
    def load_image(self, name):
        raise NotImplementedError

class ResultRepo(Repo):
    @abstractmethod
    def add_fringe(self, imgs, name):
        raise NotImplementedError
    
    @abstractmethod
    def add_image(self, imgs, name):
        raise NotImplementedError

    @abstractmethod
    def add_ref_image(self, imgs, name):
        raise NotImplementedError
    
    @abstractmethod
    def add_heightmap(self, heightmap, name):
        raise NotImplementedError
    
    @abstractmethod
    def load_fringe(self, name):
        raise NotImplementedError
    
    @abstractmethod
    def load_image(self, name):
        raise NotImplementedError

    @abstractmethod
    def load_ref_image(self, name):
        raise NotImplementedError
    
    @abstractmethod
    def load_heightmap(self, name):
        raise NotImplementedError

class CalibrationRepo(Repo):
    @abstractmethod
    def add_gamma(self, data):
        raise NotImplementedError
    
    @abstractmethod
    def add_lens(self, data):
        raise NotImplementedError
    
    @abstractmethod
    def add_proj(self, data):
        raise NotImplementedError
    
    @abstractmethod
    def load_gamma(self, cam_name):
        raise NotImplementedError
    
    @abstractmethod
    def load_lens(self, cam_name):
        raise NotImplementedError
    
    @abstractmethod
    def load_proj(self, proj_name):
        raise NotImplementedError

### CONCRETE IMPLEMENTATIONS ###

class BinRepo(Repo):
    def __init__(self, file):
        self._file = file
        
        self._outdated = True
        self._load_cache = None

        self._changes = []

    def add_bin(self, data):
        self._changes.append(data)

    def load_bin(self):
        if self._outdated:
            with open(os.path.join(self._file), 'rb') as infile:
                self._load_cache = pickle.load(infile)
                
            self._outdated = False
            
        return self._load_cache

    def commit(self):
        out = {}
        for d in self._changes:
            out = out | d
        
        with open(os.path.join(self._file), 'wb') as outfile:
            pickle.dump(out, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        self._outdated = True

class FileImageRepo(ImageRepo):
    def __init__(self, path):
        self._path = path
        self._changes = {}

    def add_image(self, img, name):
        self._changes[name] = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    def load_image(self, name):
        # BGR Format
        return cv2.imread(os.path.join(self._path, name), cv2.IMREAD_COLOR)

    def commit(self):
        # Write all images
        for name, img in self._changes.items():
            cv2.imwrite(os.path.join(self._path, name), img)

        self._changes = dict()

class BinCalibrationRepo(CalibrationRepo):
    def __init__(self, file):
        self._repo = BinRepo(file)
        
        self._data = dict()

    def add_gamma(self, data):
        cam_name = data.camera.name
        
        self.__if_ne(cam_name)
        
        self._data[cam_name]["gamma"] = data.serialize()
    
    def add_lens(self, data):
        cam_name = data.camera.name
        
        self.__if_ne(cam_name)
        
        self._data[cam_name]["lens"] = data.serialize()

    def add_proj(self, data):
        projector = data.projector.name
        
        self._data[proj_name] = data.serialize()
    
    def load_gamma(self, cam_name):
        data = self._repo.load_bin()[cam_name]
        return data["gamma"]
    
    def load_lens(self, cam_name):
        data = self._repo.load_bin()[cam_name]
        return data["lens"]
    
    def load_proj(self, proj_name):
        #data = self._repo.load_bin()[proj_name]
        return None
    
    def commit(self):
        self._repo.add_bin(self._data)
        
        self._repo.commit()
        
    def __if_ne(self, name):
        if not (name in self._data):
            self._data[name] = dict()