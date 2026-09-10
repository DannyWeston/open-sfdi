import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
# TODO: Move to .env file

import cv2
import sys

import threading
import time

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence
from enum import Enum, auto
from abc import abstractmethod

from . import DeviceSettings, DeviceBackend

from .. import utils, image, characterisation as ch

# Settings

@dataclass(frozen=True)
class CameraSettings(DeviceSettings):
    resolution: tuple[int, int] = (1280, 720)   # Pixels
    refresh_rate: float = 30.0                  # Hz
    exposure_ms: float = -1                     # -1 = autoexposure, 0< = microseconds
    focus: float = -1                           # -1 = autofocus, 0< = lens position
    # gain: Optional[float] = 1.0               # None = Not applicable, -1 = autofocus

@dataclass(frozen=True)
class OpenCVCameraSettings(CameraSettings):
    device_id: int = 0
    buffer_size: int = 1

@dataclass(frozen=True)
class PiCameraSettings(CameraSettings):
    device_id: int = 0

@dataclass(frozen=True)
class FileCameraSettings(CameraSettings):
    imgs: Sequence[image.Image] = field(default_factory=list)
    loop: bool = False

# Backends

class CameraBackend(DeviceBackend):
    def __init__(self, settings: CameraSettings):
        super().__init__(settings)

    @abstractmethod
    def open(self, settings: CameraSettings) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def is_open(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
    
    def get_settings(self) -> CameraSettings:
        return super().get_settings()

    def set_settings(self, settings: CameraSettings) -> CameraSettings:
        return super().set_settings(settings)
    
    def get_source(self):
        raise NotImplementedError

    @abstractmethod
    def load(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def process(self) -> image.Image:
        raise NotImplementedError

    @abstractmethod
    def capture(self) -> image.Image:
        raise NotImplementedError

class OpenCVCameraBackend(CameraBackend):
    def __init__(self, settings: OpenCVCameraSettings):
        super().__init__(settings=settings)

        self._handle: cv2.VideoCapture = None
        self._device_id = None

        self._exposure_table = [
            (-1, 640.0),
            (-2, 320.0),
            (-3, 160.0),
            (-4, 80.0),
            (-5, 40.0),
            (-6, 20.0),
            (-7, 10.0),
            (-8, 5.0),
            (-9, 2.5),
            (-10, 1.25),
            (-11, 0.65),
            (-12, 0.3125),
            (-13, 0.15625),
        ]

    def open(self, settings:OpenCVCameraSettings=None) -> bool:
        if self.is_open(): 
            return True

        if settings:
            settings = self.set_settings(settings)
        else:
            settings = self.get_settings()
            if settings is None:
                self.set_settings(OpenCVCameraSettings())
                settings = self.get_settings()

        if self._handle is None:
            api = cv2.CAP_ANY if sys.platform != "win32" else cv2.CAP_MSMF

            self._handle = cv2.VideoCapture(settings.device_id, apiPreference=api)

        self.set_settings(settings)

        return True

    def is_open(self):
        return (self._handle is not None) and (self._handle.isOpened())

    def get_settings(self) -> OpenCVCameraSettings:
        if not self.is_open():
            return self._initial_settings

        resolution = (int(self._handle.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._handle.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        refresh_rate = self._handle.get(cv2.CAP_PROP_FPS)
        ms = self._get_exposure_ms(self._handle.get(cv2.CAP_PROP_EXPOSURE))
        exposure_ms = -1 if (self._handle.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1) else ms
        focus = -1 if (self._handle.get(cv2.CAP_PROP_AUTOFOCUS) == 1) else self._handle.get(cv2.CAP_PROP_FOCUS)
        device_id = self._device_id

        buffer_size = self._handle.get(cv2.CAP_PROP_BUFFERSIZE)

        settings = OpenCVCameraSettings(
            resolution,
            refresh_rate,
            exposure_ms,
            focus,
            device_id,
            buffer_size
        )

        return settings

    def set_settings(self, settings: OpenCVCameraSettings) -> OpenCVCameraSettings:
        if not self.is_open():
            self._initial_settings = settings
            return settings
        
        self._device_id = settings.device_id

        # Refresh Rate
        self._handle.set(cv2.CAP_PROP_FPS, settings.refresh_rate)
        # self._handle.set(cv2.CAP_PROP_BUFFERSIZE, settings.buffer_size)

        # Resolution
        self._handle.set(cv2.CAP_PROP_FRAME_WIDTH, settings.resolution[0])
        self._handle.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.resolution[1])

        # Exposure
        if settings.exposure_ms < 0:
            self._handle.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        else:
            self._handle.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            exposure_id = self._get_exposure_id(settings.exposure_ms)
            self._handle.set(cv2.CAP_PROP_EXPOSURE, exposure_id)

        # Focus
        if settings.focus < 0:
            self._handle.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        else:
            self._handle.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self._handle.set(cv2.CAP_PROP_FOCUS, settings.focus) 

        return self.get_settings()

    def close(self):
        if self.is_open():
            self._handle.release()
            self._handle = None
            self._device_id = None

    def get_source(self):
        return CameraSource.opencv

    def _get_exposure_id(self, value):
        for i, (_, b) in enumerate(self._exposure_table):
            if b < value:
                result = self._exposure_table[max(0, i - 1)][0]
                return result

        return self._exposure_table[-1][0]

    def _get_exposure_ms(self, value):
        for i, (a, _) in enumerate(self._exposure_table):
            if a == value:
                result = self._exposure_table[i][1]
                return result

        return self._exposure_table[-1][1]

    # Camera specific

    def load(self) -> bool:
        raise NotImplementedError

    def process(self) -> image.Image:
        raise NotImplementedError

    def capture(self) -> image.Image:
        if not self.is_open():
            raise Exception("Camera has not been opened")

        try:
            ret, raw_data = self._handle.read()
        except Exception as ex:
            ret = False

        if not ret: return None

        return image.Image(image.ToInt(raw_data))

    # CV2 Specific

class PiCameraBackend(CameraBackend):
    def __init__(self, settings: PiCameraSettings):
        from picamera2 import Picamera2

        super().__init__(settings)

        self._init = False
        self._handle = Picamera2()

        self._config = self.__get_config()            
        self._handle.configure(self._config)
        self._handle.start()
    
    def is_open(self):
        return self._init

    def open(self):
        pass

    def close(self):
        if self.is_open():
            self._handle.stop()

    def get_source(self):
        return CameraSource.picamera

    def load(self) -> bool:
        raise NotImplementedError

    def process(self) -> image.Image:
        raise NotImplementedError

    def capture(self) -> image.Image:
        raw_data = self._handle.capture_array()

        # Use float (spec of program)!
        raw_data = image.ToFloat(raw_data)

        return image.Image(raw_data)


    def get_device_id(self) -> int:
        return self._device_id

    # def get_resolution(self) -> tuple[int, int]:
    #     config = self._handle.camera_configuration()

    #     return config['main']['size']
    
    # def set_resolution(self, value: tuple[int, int]):
    #     config = self._handle.create_still_configuration(
    #         main={
    #             "size": value,
    #             "format": "RGB888",
    #         },
    #     )

    #     self._handle.switch_mode(config)

class FileCameraBackend(CameraBackend):
    def __init__(self, settings: FileCameraSettings):
        super().__init__(settings)

        self._imgs: deque = None
        self._loop: bool = None

        self._resolution = (1920, 1080)
        self._refresh_rate = 30.0

        self._opened = False

    def open(self, settings: CameraSettings=None) -> bool:
        if settings:
            self.set_settings(settings)

        return True

    def is_open(self):
        return self._opened

    def close(self):
        if self.is_open():
            self._imgs.clear()
            self._opened = False

    def get_settings(self) -> CameraSettings:
        return FileCameraSettings(
            self._resolution,
            self._refresh_rate,
            -1,
            -1,
            self._imgs,
            self._loop
        )

    def set_settings(self, settings: FileCameraSettings) -> CameraSettings:
        self._resolution = settings.resolution
        self._refresh_rate = settings.refresh_rate
        self._imgs = deque(settings.imgs)
        self._loop = deque(settings.loop)

        return self.get_settings()

    def get_source(self):
        return CameraSource.filebased

    def load(self) -> bool:
        try:
            if len(self._imgs) == 0:
                return False

            if isinstance(self._imgs[0], image.FileImage):
                self._imgs[0].Preload()

        except IndexError as ex:
            return False

        return True

    def process(self) -> image.Image:
        try:
            if len(self._imgs) == 0: return None
            
            img = self._imgs.popleft()

        except IndexError as ex:
            return None

        if self.get_loop():
            self._imgs.append(img)

        return img

    def capture(self) -> image.Image:
        if not self.load(): return None

        return self.process()

class CameraSource(Enum):
    opencv = auto()
    picamera = auto()
    filebased = auto()

    @property
    def display_name(self):
        return str(self).split(".")[1]

    @classmethod
    def from_display_name(cls, display_name: str) -> Optional['CameraSource']:
        for source in cls:
            if source.display_name == display_name:
                return source
        return None

# Camera Class / Creation

class Camera(ch.ICharable):
    class CameraAsyncState(Enum):
        IDLE = "idle"
        RUNNING = "running"
        STOPPED = "stopped"

    def __init__(self, backend: CameraBackend, char:ch.ZhangChar=None):
        self._backend = backend

        self._char = ch.ZhangChar() if (char is None) else char
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
            
        self._lock = threading.Lock()
        self._async_interval = None
        self._pause_event = threading.Event()
        self._pause_event.set()

        self._capture_cbs = []
        self._configure_cbs = []
        self._char_cbs = []
        
        # State tracking
        self._state = Camera.CameraAsyncState.IDLE
        self._last_frame_time = 0.0

    def open(self, settings: CameraSettings=None):
        with self._lock:
            if self._backend.is_open():
                return True

            return self._backend.open(settings)

    def get_backend(self):
        with self._lock:
            return self._backend 

    def get_char(self) -> ch.ZhangChar:
        return self._char

    def set_char(self, value):
        self._char = value

        self._notify_char_listeners(value)

    def characterise(self, board, poi_coords, flags=0):
        resolution = self.get_settings().resolution

        char = self.get_char()
        char.execute(board, poi_coords, resolution, flags)

        self.set_char(char)

        return char

    def get_settings(self):
        backend = self.get_backend()

        with self._lock:
            settings = backend.get_settings()

        return settings

    def set_settings(self, settings: CameraSettings) -> CameraSettings:
        backend = self.get_backend()

        if not backend.is_open():
            with self._lock:
                return backend.set_settings(settings)

        # Stupid opencv bug: reload camera handle
        if self.get_async_state() == Camera.CameraAsyncState.RUNNING: 
            self.close_async()

            with self._lock:
                settings = backend.set_settings(settings)
                self._async_interval = 1.0 / settings.refresh_rate

            self.open_async()
        else:
            self.close()

            settings = backend.set_settings(settings)
            self._async_interval = 1.0 / settings.refresh_rate

            self.open()

        self._notify_configure_listeners(settings)

        return settings

    def get_async_state(self):
        return self._state
    
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def capture(self) -> image.Image:
        backend = self.get_backend()

        with self._lock:
            img = backend.capture()

        self._notify_capture_listeners(img)

        return img

    # Callbacks
    def add_capture_listener(self, func):
        self._capture_cbs.append(func)

    def remove_capture_listener(self, func):
        self._capture_cbs.remove(func)

    def _notify_capture_listeners(self, img: image.Image):
        for cb in self._capture_cbs:
            cb(img)

    def add_configure_listener(self, func):
        self._configure_cbs.append(func)

    def remove_configure_listener(self, func):
        self._configure_cbs.remove(func)

    def _notify_configure_listeners(self, settings: CameraSettings):
        for cb in self._configure_cbs:
            cb(settings)
     
    def add_char_listener(self, func):
        self._char_cbs.append(func)

    def remove_char_listener(self, func):
        self._char_cbs.remove(func)

    def _notify_char_listeners(self, char: ch.ZhangChar):
        for cb in self._char_cbs:
            cb(char)

    def open_async(self, settings: CameraSettings=None):
        state = self.get_async_state()
        if state == Camera.CameraAsyncState.RUNNING:
            return True

        if not self.open(settings):
            return False

        self._async_interval = 1.0 / self.get_settings().refresh_rate

        self._stop_event.clear()
        self._pause_event.set()  # Start unpaused
        
        self._thread = threading.Thread(
            target=self._loop_async,
            daemon=True,
        )
        self._thread.start()
        self._state = Camera.CameraAsyncState.RUNNING

        return True

    def pause_async(self):
        self._pause_event.clear()

    def resume_async(self):
        self._pause_event.set()
        self._last_frame_time = time.time()

    def close_async(self):
        with self._lock:
            if self._state == Camera.CameraAsyncState.STOPPED:
                return

            self._stop_event.set()
            self._pause_event.set()
            self._state = Camera.CameraAsyncState.STOPPED
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)

        self.close()

    def _loop_async(self):
        while not self._stop_event.is_set():
            if not self._pause_event.is_set():
                time.sleep(0.001)
                continue
            
            current_time = time.time()
            elapsed = current_time - self._last_frame_time
            
            if elapsed < self._async_interval:
                sleep_time = self._async_interval - elapsed
                time.sleep(min(sleep_time, 0.01))
                continue

            img = self.capture()

            if img is None:
                continue

            self._last_frame_time = time.time()

            self._notify_capture_listeners(img)

    # Misc
    def close(self):
        self.close_async()

        self._backend.close()

    def __enter__(self):
        self.open_async()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_async()

    def __str__(self):
        char = self.get_char()

        v = "<Camera>"
        v += f" {char}" if char.is_characterised else " (Not Characterised)"
        return v

class CameraFactory:
    @staticmethod
    def make_camera(source, settings: CameraSettings=None, char: ch.ZhangChar=None):
        match source:
            case CameraSource.opencv:
                backend = OpenCVCameraBackend(settings)

            case CameraSource.picamera:
                backend = PiCameraBackend(settings)

            case CameraSource.filebased:
                backend = FileCameraBackend(settings)

        return Camera(backend, char)

    @staticmethod
    def make_settings(source, params):
        match source:
            case CameraSource.opencv:
                return OpenCVCameraSettings(**params)

            case CameraSource.picamera:
                return PiCameraSettings(**params)

            case CameraSource.filebased:
                return FileCameraSettings(**params)

        return None