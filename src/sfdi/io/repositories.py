import cv2
import logging
import os

from abc import ABC, abstractmethod

from sfdi.definitions import RESULTS_DIR

from datetime import datetime

class Repository(ABC):
    def __init__(self):
        self.logger = logging.getLogger('sfdi')
    
    @abstractmethod
    def save(self, value):
        raise NotImplementedError
    
    @abstractmethod
    def load(self, value):
        raise NotImplementedError
    
class ImageRepository(Repository):
    def __init__(self, path, ext=".jpg"):
        super().__init__()
        
        self.path = path
        self.ext = ext
    
    def save(self, img, name=None):
        name = f'{name}{self.ext}' if name else f'{datetime.now().strftime("%Y%m%d_%H%M%S")}{self.ext}'
        cv2.imwrite(os.path.join(self.path, name), img)
        return name
    
    def load(self, name):
        return cv2.imread(os.path.join(self.path, name), 1)

class ResultRepository(Repository):
    """
        Saves some results (treated as JsonObject) subdirectory inside RESULTS_DIR with a
        given name. Optionally, any passed in images can be saved to the same directory.

        Args:
            name (str): Name of the directory to be created.
            results (dict): collection of results (representing JSON).
            ref_imgs (list): Optional - collected reference images to save.
            imgs (list): Optional - collected images to save.

        Returns:
            None
    """
    def __init__(self, path=RESULTS_DIR):
        super().__init__()
        self.path = path
        
        self.directory = os.path.join(path, str(datetime.now().strftime("%Y%m%d_%H%M%S")))
        os.mkdir(self.directory, 0o770)

    def save(self, expresult):
        if expresult.results: # Save numerical results
            with open(os.path.join(self.directory, 'results.json'), 'w') as outfile:
                json.dump(expresult.results, outfile, indent=4)

        if expresult.fringes: # Save used fringe patterns
            for i, img in enumerate(expresult.fringes):
                cv2.imwrite(os.path.join(self.directory, f'fringes{i}.jpg'), img)

        if expresult.imgs: # Save recorded images for each camera
            for i, imgs in enumerate(expresult.imgs):
                for cam_i, img in enumerate(imgs):
                    cv2.imwrite(os.path.join(self.directory, f'cam{cam_i}_img{i}.jpg'), img)
            
        if expresult.ref_imgs: # Save recorded reference images for each camera
            for i, imgs in enumerate(expresult.ref_imgs):
                for cam_i, img in enumerate(imgs):
                    cv2.imwrite(os.path.join(self.directory, f'cam{cam_i}_refimg{i}.jpg'), img)
                    
        self.logger.info(f"Results saved in {self.directory}")
                    
        return self.directory

    def load(self):
        raise NotImplementedError