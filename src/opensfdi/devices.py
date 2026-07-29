import cv2
import threading
import sys
import queue
import time

from typing import Optional
from enum import Enum
from importlib import util
from abc import abstractmethod

from . import utils, image, characterisation as ch

# Cameras

class CameraState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"

class BaseCamera(utils.SerialisableMixin, ch.ICharable):
    def __init__(self, resolution: tuple[int, int], refresh_rate: float, exposure:int=0, auto_exposure:float=0, char:ch.ZhangChar=None):

        self._char = ch.ZhangChar() if (char is None) else char

        self._resolution = resolution
        self._refresh_rate = refresh_rate
        self._exposure = exposure
        self._auto_exposure = auto_exposure
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
            
        self._lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()

        self._capture_cbs = []
        self._exposure_cbs = []
        self._auto_exposure_cbs = []
        self._refresh_rate_cbs = []
        self._resolution_cbs = []
        self._char_cbs = []
        
        # State tracking
        self._state = CameraState.IDLE
        self._last_frame_time = 0.0

    @property
    def exclude_fields(self):
        return super().exclude_fields.union({
            "_thread", "_lock", "_stop_event", "_pause_event",
            
            "_capture_cbs", "_exposure_cbs", "_auto_exposure_cbs",
            "_refresh_rate_cbs", "_resolution_cbs","_char_cbs",
            
            "_state", "_last_frame_time"
        })

    def get_char(self):
        with self._lock:
            return self._char
    
    def set_char(self, value):
        with self._lock:
            if self._char != value:
                self._char = value
                
                for cb in self._char_cbs: cb(value)

    def get_resolution(self) -> tuple[int, int]:
        with self._lock:
            return self._resolution
    
    def set_resolution(self, value: tuple[int, int]):
        with self._lock:
            if self._resolution != value:
                self._resolution = value

                for cb in self._resolution_cbs:
                    cb(*value)

    def get_refresh_rate(self) -> float:
        with self._lock:
            return self._refresh_rate
    
    def set_refresh_rate(self, value: float):
        with self._lock:
            if self._refresh_rate != value:
                self._refresh_rate = value

                for cb in self._refresh_rate_cbs: cb(value)

    def get_auto_exposure(self):
        with self._lock:
            return self._auto_exposure

    def set_auto_exposure(self, value: float):
        with self._lock:
            if self._auto_exposure != value:
                self._auto_exposure = value
                
                for cb in self._auto_exposure_cbs: cb(value)

    def get_exposure(self) -> int:
        with self._lock:
            return self._exposure
            
    def set_exposure(self, value: int):
        with self._lock:
            if self._exposure != value:
                self._exposure = value
                
                for cb in self._exposure_cbs: cb(value)
    
    def get_camera_state(self):
        return self._state
    
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()
    

    # Methods
    @abstractmethod
    def _init_async(self):
        raise NotImplementedError

    @abstractmethod
    def _capture(self) -> image.Image:
        raise NotImplementedError


    # Callbacks
    def add_capture_listener(self, func):
        self._capture_cbs.append(func)

    def remove_capture_listener(self, func):
        self._capture_cbs.remove(func)

    def _notify_capture_listeners(self, img: image.Image):
        with self._lock:
            for cb in self._capture_cbs:
                cb(img)

    def add_exposure_listener(self, func):
        self._exposure_cbs.append(func)

    def add_auto_exposure_listener(self, func):
        self._auto_exposure_cbs.append(func)

    def add_refresh_rate_listener(self, func):
        self._refresh_rate_cbs.append(func)

    def add_resolution_listener(self, func):
        self._resolution_cbs.append(func)

    def add_char_listener(self, func):
        self._char_cbs.append(func)

    # Async loop
    def _async_loop(self):
        while not self._stop_event.is_set():
            if not self._pause_event.is_set():
                time.sleep(0.001)
                continue

            interval = 1.0 / self.get_refresh_rate()
            
            current_time = time.time()
            elapsed = current_time - self._last_frame_time
            
            if elapsed < interval:
                sleep_time = interval - elapsed
                time.sleep(min(sleep_time, 0.01))
                continue

            img = self._capture()

            if img is None: 
                self._state = CameraState.ERROR
                time.sleep(0.01)
                continue

            self._last_frame_time = time.time()

            try:
                self._notify_capture_listeners(img)
            except queue.Full:
                pass

    def start_async(self) -> bool:
        with self._lock:
            state = self.get_camera_state()
            if state == CameraState.RUNNING:
                return True
                
            if state == CameraState.ERROR:
                self.cleanup()
                return False
            
            self._init_async()

            self._stop_event.clear()
            self._pause_event.set()  # Start unpaused
            
            self._thread = threading.Thread(
                target=self._async_loop,
                daemon=True,
            )
            self._thread.start()
            self._state = CameraState.RUNNING
            
            return True
   
    def pause_async(self):
        self._pause_event.clear()
    
    def resume_async(self):
        self._pause_event.set()
        self._last_frame_time = time.time()
    
    def stop_async(self):
        with self._lock:
            if self._state == CameraState.STOPPED:
                return
                
            self._stop_event.set()
            self._pause_event.set()
            self._state = CameraState.STOPPED
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)

    # Blocking
    def read(self) -> Optional[image.Image]:
        img = self._capture()

        try:
            self._notify_capture_listeners(img)
        except queue.Full:
            pass

        return img

    # Misc
    def cleanup(self):
        self.stop_async()

        self._capture_cbs = []

    def __enter__(self):
        self.start_async()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_async()

    def __str__(self):
        v = "<Camera>"
        v += f" {self.char}" if self.char else " (Not Characterised)"
        return v

class OpenCVCamera(BaseCamera):
    def __init__(self, resolution: tuple[int, int], refresh_rate: float, device_id:int, exposure:int=0, auto_exposure:float=0, char:ch.ZhangChar=None):
        super().__init__(resolution, refresh_rate, exposure, auto_exposure, char=char)

        self._device_id = device_id

        self._api = cv2.CAP_ANY if sys.platform != "win32" else cv2.CAP_MSMF
        self._camera_handle = cv2.VideoCapture(self._device_id, apiPreference=self._api)

        self._init_cv_props()

    
    @property
    def exclude_fields(self):
        return super().exclude_fields.union({
            '_camera_handle', "_api"
        })

    def _capture(self) -> image.Image:
        ret, raw_data = self._camera_handle.read()

        if not ret:
            raise Exception("Could not capture an image with the CV2Camera")

        return image.Image(image.ToInt(raw_data))

    def _init_async(self) -> bool:
        if not self._camera_handle.isOpened():
            self._state = CameraState.ERROR
            return False
        
        return True

    def _init_cv_props(self):
        self._camera_handle.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._camera_handle.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self._camera_handle.set(cv2.CAP_PROP_FRAME_WIDTH, self.get_resolution()[0])
        self._camera_handle.set(cv2.CAP_PROP_FRAME_HEIGHT, self.get_resolution()[1])
        self._camera_handle.set(cv2.CAP_PROP_FPS, self.get_refresh_rate())
        self._camera_handle.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # TODO: Investigate weird thing with 0.25 autoexposure = off?
        self._camera_handle.set(cv2.CAP_PROP_EXPOSURE, self.get_exposure())
        self._camera_handle.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.get_auto_exposure())


    @property
    def device_id(self):
        with self._lock:
            return self._device_id

    def set_resolution(self, value: tuple[int, int]):
        super().set_resolution(value)

        if self._camera_handle:
            self._camera_handle.set(cv2.CAP_PROP_FRAME_WIDTH, value[0])
            self._camera_handle.set(cv2.CAP_PROP_FRAME_HEIGHT, value[1])

    def set_refresh_rate(self, value: float):
        super().set_refresh_rate(value)

        if self._camera_handle:
            self._camera_handle.set(cv2.CAP_PROP_FPS, value)
    
    def set_exposure(self, value: float):
        super().set_exposure(value)

        self._init_cv_props()

    def set_auto_exposure(self, value: float):
        super().set_auto_exposure(value)

        self._init_cv_props()

    def cleanup(self):
        super().cleanup()

        if self._camera_handle.isOpened():
            self._camera_handle.release()

class FileCamera(BaseCamera):
    _exclude_fields = {
        '_images', 
        '_prefetch', '_preloaded_count', '_stop_event', '_prefetcher_thread', '_lock', '_xp'
    }

    def __init__(self, resolution: tuple[int, int], refresh_rate: float, exposure=0, auto_exposure=0, images:list[image.FileImage]=None, prefetch=-1, 
        char:ch.ZhangChar=None
    ):
        super().__init__(resolution, refresh_rate, exposure, auto_exposure, char=char)

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

    def _capture(self) -> image.Image:
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

    def _init_async(self):
        pass

# Raspberry Pi based camera

if util.find_spec("picamera2"):
    from picamera2 import Picamera2

    class PiCamera(BaseCamera):
        _exclude_fields = {'_camera_handle'}

        def __init__(self, resolution:tuple[int, int], refresh_rate:float, device_id:int, exposure=0, auto_exposure=0, char:ch.ZhangChar=None):
            super().__init__(resolution, refresh_rate, exposure, auto_exposure, char=char)

            # Capture an image
            self._init = False
            self._camera_handle = None
            # self._camera_handle.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            
            self._device_id = device_id

            # Picamera2 Handle
            self._handle = Picamera2()

            self._config = self.__get_config()            
            self._handle.configure(self._config)
            self._handle.start()

        def _init_async(self):
            pass

        def __get_config(self):
            return self._handle.create_still_configuration(
                main={
                    "size": self.resolution,
                    "format": "RGB888",
                },
                controls={
                    "FrameRate" : self.refresh_rate
                }
            )

        def __update_config(self):
            self._config = self.__get_config()

            self._handle.switch_mode(self._config)
            
        @property
        def resolution(self) -> tuple[int, int]:
            return super().resolution
        
        @resolution.setter
        def resolution(self, value: tuple[int, int]):
            self._resolution = value

            self.__update_config()

        @property
        def refresh_rate(self) -> float:
            return super().refresh_rate

        @refresh_rate.setter
        def refresh_rate(self, value: float):
            self._refresh_rate = value

            self.__update_config()

        def _capture(self) -> image.Image:
            raw_data = self._handle.capture_array()

            # Use float (spec of program)!
            raw_data = image.ToFloat(raw_data)

            return image.Image(raw_data)
        
        def cleanup(self):
            if self._handle:
                self._handle.stop()

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
    
    def cleanup(self):
        pass
    
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

        captured_imgs.append(camera._capture().raw_data)

    return xp.asarray(captured_imgs)

    # Find first observable change of values for averages (left and right sides) i.e >= delta
    # s, f = DetectableIndices(averages, delta)

    # vis_averages = averages[s:f+1]
    # vis_intensities = intensities[s:f+1]

    # coeffs = xp.polyfit(vis_averages, vis_intensities, order)
    # visible = intensities[s:f+1]

    # return coeffs, visible