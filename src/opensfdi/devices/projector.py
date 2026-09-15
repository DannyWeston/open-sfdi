import threading
import queue
import time

from dataclasses import dataclass

from enum import Enum, auto
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QSizePolicy, QLabel

from . import DeviceSettings, DeviceBackend
from .. import utils, image, characterisation as ch

# Settings

@dataclass(frozen=True)
class ProjectorSettings(DeviceSettings):
    resolution: tuple[int, int] = (1280, 720)   # Pixels
    refresh_rate: float = 30.0                  # Hz
    # gain: Optional[float] = 1.0                 # None = Not applicable, -1 = autofocus

@dataclass(frozen=True)
class StubProjectorSettings(ProjectorSettings):
    pass

@dataclass(frozen=True)
class PysideProjectorSettings(ProjectorSettings):
    display_id: int = 0
    app = None

class ProjectorBackend(DeviceBackend):
    def __init__(self, settings: ProjectorSettings):
        super().__init__(settings)

    def open(self, settings: ProjectorSettings) -> bool:
        raise NotImplementedError

    def is_open(self) -> bool:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def get_settings(self) -> ProjectorSettings:
        return super().get_settings()

    def set_settings(self, settings: ProjectorSettings) -> ProjectorSettings:
        return super().set_settings(settings)

    # Projector specific settings
    
    def preload(self) -> bool:
        raise NotImplementedError

    def process(self) -> image.Image:
        raise NotImplementedError

    def project(self, img: image.Image):
        raise NotImplementedError

class StubProjectorBackend(ProjectorBackend):
    def __init__(self, settings: StubProjectorSettings):
        super().__init__(settings)

        self._open = False
        self._initial_settings = ProjectorSettings()

    def open(self, settings: StubProjectorSettings) -> bool:
        if self.is_open(): return True

        if settings:
            self.set_settings(settings)

            self._open = True

        return True

    def is_open(self) -> bool:
        return True

    def cleanup(self):
        pass

    def get_settings(self) -> StubProjectorSettings:
        return super().get_settings()

    def set_settings(self, settings: StubProjectorSettings) -> StubProjectorSettings:
        return super().set_settings(settings)
    
    def preload(self) -> bool:
        pass

    def process(self) -> image.Image:
        pass

    def project(self, img: image.Image):
        pass



    # Projector specific settings

    def preload(self) -> bool:
        raise NotImplementedError

    def process(self) -> image.Image:
        raise NotImplementedError

    def project(self, img: image.Image):
        raise NotImplementedError

class PysideProjectorBackend(ProjectorBackend):
    class ProjectorWidget(QLabel):
        def __init__(self, *args, **kwargs):
            super().__init__()

            self.setStyleSheet("""
                QLabel {
                    border: none;
                    background-color: transparent;
                }
            """)

            self._pixmap = self.pixmap()
            self._resized = False

        def resizeEvent(self, event):
            self.setPixmap(self._pixmap)

        def setPixmap(self, pixmap: QPixmap):
            if not pixmap: return

            self._pixmap = pixmap

            new_pixmap = self._pixmap.scaled(self.frameSize(), Qt.AspectRatioMode.KeepAspectRatio)

            return QLabel.setPixmap(self, new_pixmap)

        def to_pixmap(self, img: image.Image):
            with utils.ProcessingContext.UseGPU(False):
                xp = utils.ProcessingContext().xp
                data = image.ToInt(img.raw_data) # Use uint8 format
                data = xp.asarray(data)

                if data.ndim == 3: q_img = QImage(data, data.shape[1], data.shape[0], QImage.Format.Format_BGR888)

                elif data.ndim == 2: q_img = QImage(data, data.shape[1], data.shape[0], QImage.Format.Format_Grayscale8)

                return q_img
    
        def set_image(self, img: QImage):
            self.setPixmap(QPixmap.fromImage(img))

        def clear(self, w=400, h=400):
            empty = QPixmap(w, h)
            empty.fill(Qt.GlobalColor.black)
            self.setPixmap(empty)

    def __init__(self, settings: PysideProjectorSettings):
        super().__init__(settings)

        self._app = None
        self._frame_drawn = None
        self._img = None
        self._display_id = None

        self._resolution = None
        self._refresh_rate = None

    def open(self, settings: PysideProjectorSettings) -> bool:  
        if self.is_open(): return True

        self._app = settings.app
        self._display_id = settings.display_id

        self._window = QMainWindow(parent=None)

        self._window.setWindowFlags(
            Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.NoDropShadowWindowHint |
            Qt.WindowType.WindowTransparentForInput |
            Qt.WindowType.NoTitleBarBackgroundHint
        )
        self._window.setContentsMargins(0, 0, 0, 0)
    
        self._window.setFocus()
        self._window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # self._window.setCursor(Qt.CursorShape.BlankCursor)

        display = QApplication.screens()[self.get_display_id()]
        self._window.setScreen(display)
        self._window.move(display.geometry().topLeft())

        self.widget_projector = PysideProjectorBackend.ProjectorWidget(self._window)

        self.widget_projector.setContentsMargins(0, 0, 0, 0)
        self._window.setCentralWidget(self.widget_projector)
        self.widget_projector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.widget_projector.setStyleSheet("""
            QLabel {
                border: none;
                background-color: red;
            }
        """)

        # TODO: Incorporate settings

        # Default img white
        self._window.showMaximized()
        self._window.showFullScreen()
        self._window.activateWindow()

        return True

    def is_open(self) -> bool:
        return True

    def close(self):
        self.widget_projector.hide()
        self._window.deleteLater()

        self._app = None
        self._display_id = None

    def get_settings(self) -> PysideProjectorSettings:
        return PysideProjectorSettings(
            self._resolution,
            self._refresh_rate,
            self._display_id,
            self._app
        )

    def set_settings(self, settings: PysideProjectorSettings) -> PysideProjectorSettings:
        self._initial_settings = settings
        
        self._resolution = settings.resolution
        self._refresh_rate = settings.refresh_rate

        self._app = settings.app
        self._display_id = settings.display_id
        self._refresh_rate = settings.refresh_rate

        return self.get_settings()

    # Projector methods

    def preload(self, img: image.Image) -> bool:
        self._img = self.widget_projector.to_pixmap(img)

        return True

    def process(self):
        self.widget_projector.set_image(self._img)

    def project(self, img: image.Image):
        if self.preload(img):
            self.process()

    # def _check_quit(self):
    #     if len(QApplication.topLevelWindows()) == 1: # Only this window left
    #         self.cleanup()
    #         self._window.deleteLater()

@dataclass(frozen=True)
class ProjectorSource(Enum):
    stub = auto()
    pyside6 = auto()

    @property
    def display_name(self):
        return str(self).split(".")[1]

    @classmethod
    def from_display_name(cls, display_name: str) -> Optional['ProjectorSource']:
        for source in cls:
            if source.display_name == display_name:
                return source
        return None


# Projector class and creation

class Projector(ch.ICharable, ProjectorBackend):
    class ProjectorAsyncState(Enum):
        IDLE = "idle"
        RUNNING = "running"
        STOPPED = "stopped"

    def __init__(self, backend: ProjectorBackend, char:ch.ZhangChar=None):
        self._backend = backend        
        self._char = ch.ZhangChar() if (char is None) else char

        self._lock = threading.Lock()

        # self._thread: Optional[threading.Thread] = None
        # self._async_imgs = queue.Queue()

        # self._stop_event = threading.Event()
        # self._pause_event = threading.Event()
        # self._pause_event.set()

        self._project_cbs = []
        self._configure_cbs = []
        self._char_cbs = []

        # State tracking
        # self._state = Projector.ProjectorAsyncState.IDLE
        self._last_frame_time = 0.0

    def open(self, settings: ProjectorSettings=None):
        if self._backend.is_open():
            return True

        return self._backend.open(settings)

    def get_backend(self):
        with self._lock:
            return self._backend 

    def get_char(self):
        with self._lock:
            return self._char
    
    def set_char(self, value):
        with self._lock:
            self._char = value
            
            self._notify_char_listeners(value)

    def characterise(self, board, poi_coords, hint: ch.ZhangCharHint, extra_flags=0):
        resolution = self.get_settings().resolution

        char = ch.ZhangChar.characterise(
            board, poi_coords, resolution,
            hint, extra_flags
        )

        self.set_char(char)

        return char

    def get_settings(self) -> ProjectorSettings:
        return self.get_backend().get_settings()

    def set_settings(self, settings: ProjectorSettings) -> ProjectorSettings:
        settings = self.get_backend().set_settings(settings)

        self._notify_configure_listeners(settings)

        return settings

    def get_async_state(self):
        return self._state

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def project(self, img: image.Image):
        img = self.get_backend().project(img)

        try:
            self._notify_project_listeners(img)
        except queue.Full:
            return False

        return True

    def async_project(self, img: image.Image):
        self._async_imgs.put_nowait(img)

    # Callbacks
    def add_project_listener(self, func):
        self._project_cbs.append(func)

    def remove_project_listener(self, func):
        self._project_cbs.remove(func)

    def _notify_project_listeners(self, img: image.Image):
        for cb in self._project_cbs:
            cb(img)

    def add_configure_listener(self, func):
        self._configure_cbs.append(func)

    def remove_configure_listener(self, func):
        self._configure_cbs.remove(func)

    def _notify_configure_listeners(self, settings: ProjectorSettings):
        for cb in self._configure_cbs:
            cb(settings)
     
    def add_char_listener(self, func):
        self._char_cbs.append(func)

    def remove_char_listener(self, func):
        self._char_cbs.remove(func)

    def _notify_char_listeners(self, char: ch.ZhangChar):
        for cb in self._char_cbs:
            cb(char)


    def start_async(self):
        with self._lock:
            state = self.get_async_state()
            if state == Projector.ProjectorAsyncState.RUNNING:
                return
            
            self._init_async()

            self._stop_event.clear()
            self._pause_event.set()  # Start unpaused
            
            self._thread = threading.Thread(
                target=self._async_loop,
                daemon=True,
            )
            self._thread.start()
            self._state = Projector.ProjectorAsyncState.RUNNING

    def pause_async(self):
        self._pause_event.clear()
    
    def resume_async(self):
        self._pause_event.set()
        self._last_frame_time = time.time()

    def stop_async(self):
        with self._lock:
            if self._state == Projector.ProjectorAsyncState.STOPPED:
                return

            self._stop_event.set()
            self._pause_event.set()
            self._state = Projector.ProjectorAsyncState.STOPPED
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)

    def _init_async(self) -> bool:
        with self._lock:
            if not self.open():
                return False

            self._async_imgs = queue.Queue()
            
            return True

    def _async_loop(self):
        while not self._stop_event.is_set():
            if not self._pause_event.is_set():
                time.sleep(0.001)
                continue

            interval = 1.0 / self.get_settings().refresh_rate
            
            current_time = time.time()
            elapsed = current_time - self._last_frame_time
            
            if elapsed < interval:
                sleep_time = interval - elapsed
                time.sleep(min(sleep_time, 0.01))
                continue

            try: img = self._async_imgs.get_nowait()
            except: img = None

            self.project(img)

            if img is None:
                continue

            self._last_frame_time = time.time()

            self._notify_project_listeners(img)


    def cleanup(self):
        pass

    def __str__(self):
        v = "<Projector>"

        char = self.get_char()

        v += f" {char}" if char else " (Not Characterised)"

        return v

class ProjectorFactory:
    @staticmethod
    def make_projector(source, settings: ProjectorSettings, char: ch.ZhangChar=None):
        match source:
            case ProjectorSource.stub:
                new_settings = StubProjectorSettings(
                    settings.resolution,
                    settings.refresh_rate
                )

                backend = StubProjectorBackend(new_settings)

            case ProjectorSource.pyside6:
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if app is None:
                    app = QApplication([])

                new_settings = PysideProjectorSettings(
                    settings.resolution,
                    settings.refresh_rate,
                )

                backend = PysideProjectorBackend()

        return Projector(backend, char)