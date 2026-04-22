import cv2
import threading
from importlib import util

from abc import abstractmethod

from . import utils, image, characterisation as ch

# Cameras

class BaseCamera(utils.SerialisableMixin, ch.ICharable):
    def __init__(self, resolution: tuple[int, int], refresh_rate: float, 
                 exposure:int=0, auto_exposure=False, char:ch.ZhangChar=None):
        
        self._char = char

        self._resolution = resolution
        self._refresh_rate = refresh_rate
        self._exposure = exposure
        self._auto_exposure = auto_exposure

    @property
    def auto_exposure(self):
        return self._auto_exposure

    @property
    def char(self):
        return self._char
    
    @char.setter
    def char(self, value):
        self._char = value

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def refresh_rate(self) -> float:
        return self._refresh_rate

    @property
    def exposure(self) -> int:
        return self._exposure

    @property
    def auto_exposure(self) -> bool:
        return self._auto_exposure

    @abstractmethod
    def capture(self) -> image.Image:
        """ Capture an image NOTE: should return Image object"""
        raise NotImplementedError

    # Optional
    def cleanup(self):
        pass

    def __str__(self):
        v = "<Camera>"
        v += f" {self.char}" if self.char else " (Not Characterised)"
        return v

class OpenCVCamera(BaseCamera):
    _exclude_fields = {'_camera_handle'}

    def __init__(self, resolution: tuple[int, int], refresh_rate: float, 
                  exposure:int=0, auto_exposure=False, device_id:int=0, char:ch.ZhangChar=None):
        super().__init__(resolution, refresh_rate, exposure=exposure, auto_exposure=auto_exposure, char=char)

        self._device_id = device_id
        self._camera_handle = cv2.VideoCapture(device_id, apiPreference=cv2.CAP_DSHOW)
        self._set_cv_props()

    def _set_cv_props(self):
        self._camera_handle.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._camera_handle.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.resolution[0]))
        self._camera_handle.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.resolution[1]))
        self._camera_handle.set(cv2.CAP_PROP_FPS, float(self.refresh_rate))

        print(self._camera_handle.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        print(self._camera_handle.get(cv2.CAP_PROP_EXPOSURE))

        # self._camera_handle.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1 if self.auto_exposure else 1)

        # if not self.auto_exposure:
            # self._camera_handle.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

    @property
    def device_id(self):
        return self._device_id

    @device_id.setter
    def device_id(self, value:int):
        self._device_id = value

        self._camera_handle = cv2.VideoCapture(value, apiPreference=cv2.CAP_DSHOW)
        self._set_cv_props()

    @property
    def resolution(self):
        return super().resolution

    @resolution.setter
    def resolution(self, value: tuple[int, int]):
        self._resolution = value

        if self._camera_handle:
            self._camera_handle.set(cv2.CAP_PROP_FRAME_WIDTH, value[0])
            self._camera_handle.set(cv2.CAP_PROP_FRAME_HEIGHT, value[1])

    @property
    def refresh_rate(self):
        return super().refresh_rate

    @refresh_rate.setter
    def refresh_rate(self, value: float):
        super().refresh_rate = value

        self._camera_handle.set(cv2.CAP_PROP_FPS, value)

    @property
    def exposure(self):
        return super().exposure
    
    @exposure.setter
    def exposure(self, value):
        if value is None:
            self._camera_handle.set(cv2.CAP_PROP_EXPOSURE, value)
        else:
            super().exposure = value

    @property
    def auto_exposure(self) -> bool:
        return super().auto_exposure
    
    @auto_exposure.setter
    def auto_exposure(self, value: bool):
        super().auto_exposure = value

        self._camera_handle.set(cv2.CV_CAP_PROP_AUTO_EXPOSURE, 1 if value else 0)

    def capture(self) -> image.Image:
        ret, raw_data = self._camera_handle.read()

        if not ret:
            # Couldn't capture image, throw exception
            raise Exception("Could not capture an image with the CV2Camera")
        
        # Use float (spec of program)!
        raw_data = image.ToFloat(raw_data)

        # Undistort if can
        # rawImg = self.characterisation.Undistort(rawImg)

        return image.Image(raw_data)
    
    def cleanup(self):
        if self._camera_handle.isOpened():
            self._camera_handle.release()

class FileCamera(BaseCamera):
    _exclude_fields = {
        '_images', 
        '_prefetch', '_preloaded_count', '_stop_event', '_prefetcher_thread', '_lock', '_xp'
    }

    def __init__(self, resolution: tuple[int, int], refresh_rate: float, exposure:int=0, auto_exposure=False, images:list[image.FileImage]=None, prefetch=-1, 
        char:ch.ZhangChar=None
    ):
        super().__init__(resolution, refresh_rate, exposure=exposure, auto_exposure=auto_exposure, char=char)

        self._images = images

        self._prefetch = prefetch
        self._preloaded_count = 0

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._prefetcher_thread = None

    @property
    def images(self) -> list[image.FileImage]:
        return self._images

    @images.setter
    def images(self, value):
        with self._lock:
            self._images = value

            # Check if prefetching enabled
            if self._prefetch < 0: return

            self._preloaded_count = 0

            self.start_batch()

    def _prefetcher_worker(self):
        """Background thread that preloads images"""
        while not self._stop_event.is_set():
            with self._lock:
                if 0 < len(self.images) and self._preloaded_count < self._prefetch:
                    print(f"Preloaded: {self._preloaded_count}")
                    self._images[self._preloaded_count].Preload()
                    self._preloaded_count += 1
                    print("Loaded image")
                else:
                    self._stop_event.set()

    def _worker_inactive(self) -> bool:
        return (self._prefetcher_thread is None) or (not self._prefetcher_thread.is_alive())

    def start_batch(self):
        self._stop_event.clear()
        self._prefetcher_thread = threading.Thread(target=self._prefetcher_worker)
        self._prefetcher_thread.daemon = True # Thread dies with main
        self._prefetcher_thread.start()

    def capture(self) -> image.Image:
        with self._lock:
            try:
                img = self._images.pop(0)

                if self._prefetch < 0:
                    return img
                
                if img._preloaded: # Removed preloaded image
                    self._preloaded_count -= 1

                else: # Preload miss
                    print("Preload miss")
                    if self._worker_inactive():
                        self.start_batch()

                return img

            except IndexError:
                return None

# Raspberry Pi based camera

if util.find_spec("picamera2"):
    class PiCamera(BaseCamera):
        _exclude_fields = {'_camera_handle'}

        def __init__(self, resolution:tuple[int, int], refresh_rate:float, device_id:int = 0, char:ch.ZhangChar=None):
            super().__init__(resolution, refresh_rate, char=char)

            # Capture an image
            self._init = False
            self._camera_handle = None
            # self._camera_handle.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            
            self._device_id = device_id
        
        @resolution.setter
        def resolution(self, value: tuple[int, int]):
            self._resolution = value

            # self._camera_handle.set(cv2.CAP_PROP_FRAME_WIDTH, value[0])
            # self._camera_handle.set(cv2.CAP_PROP_FRAME_HEIGHT, value[1])

        @refresh_rate.setter
        def refresh_rate(self, value: float):
            self._refresh_rate = value

            # self._camera_handle.set(cv2.CAP_PROP_FPS, value)

        def capture(self) -> image.Image:
            raw_data = None

            # Use float (spec of program)!
            raw_data = image.ToFloat(raw_data)

            return image.Image(raw_data)
        
        def cleanup(self):
            if self._camera_handle:
                pass

# Projectors

class BaseProjector(utils.SerialisableMixin, ch.ICharable):
    _exclude_fields = {'_debug', '_should_undistort'}

    @abstractmethod
    def __init__(self, char:ch.ZhangChar=None):
        self._char = char

        self._should_undistort = True
        self._debug = False

    @property
    def char(self) -> ch.ZhangChar:
        return self._char

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def refresh_rate(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def throw_ratio(self) -> float:
        raise NotImplementedError
    
    @property
    def aspect_ratio(self):
        return self.resolution[0] / self.resolution[1]

    @property
    def debug(self):
        return self._debug
    
    @debug.setter
    def debug(self, value):
        self._debug = value

    @abstractmethod
    def display(self, img: image.Image):
        raise NotImplementedError
    
    def __str__(self):
        v = "<Projector>"

        if self.char:
            v += f" {self.char}"

        else: v += " (Not Characterised)"

        return v

# Utility methods

def gather_gamma_imgs(camera: BaseCamera, projector: BaseProjector, intensities):
    xp = utils.ProcessingContext().xp

    captured_imgs = []

    for intensity in intensities:
        project_img = xp.ones(projector.shape, dtype=xp.float32) * intensity

        # TODO: Callbacks
        projector.display(project_img)

        captured_imgs.append(camera.capture().raw_data)

    return xp.asarray(captured_imgs)

    # Find first observable change of values for averages (left and right sides) i.e >= delta
    # s, f = DetectableIndices(averages, delta)

    # vis_averages = averages[s:f+1]
    # vis_intensities = intensities[s:f+1]

    # coeffs = xp.polyfit(vis_averages, vis_intensities, order)
    # visible = intensities[s:f+1]

    # return coeffs, visible