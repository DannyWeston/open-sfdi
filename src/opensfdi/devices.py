import cv2
import threading

from abc import abstractmethod

from . import utils, image, characterisation as ch

# Cameras

class BaseCamera(utils.SerialisableMixin, ch.ICharable):
    def __init__(self, char:ch.ZhangChar=None):
        self._char = char

    @property
    def char(self):
        return self._char

    @property
    @abstractmethod
    def channels(self) -> int:
        raise NotImplementedError

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
    def shape(self):
        raise NotImplementedError

    @abstractmethod
    def capture(self) -> image.Image:
        """ Capture an image NOTE: should return Image object"""
        raise NotImplementedError

    def __str__(self):
        v = "<Camera>"

        if self.char:
            v += f" {self.char}"

        else: v+= " (Not Characterised)"

        return v

class CV2Camera(BaseCamera):
    _exclude_fields = {'_camera_handle'}

    def __init__(self, device_id: int, resolution: tuple[int, int], channels: int, refresh_rate: float, 
        char:ch.ZhangChar=None
    ):
        super().__init__(char=char)

        # Capture an image
        self._camera_handle = cv2.VideoCapture(device_id)

        self._channels = channels
        self._refresh_rate = refresh_rate

        # Resolution
        self._camera_handle.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self._camera_handle.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self._camera_handle.set(cv2.CAP_PROP_FPS, refresh_rate)

    @property
    def resolution(self) -> tuple[int, int]:
        return (
            int(self._camera_handle.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._camera_handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    
    @resolution.setter
    def resolution(self, value: tuple[int, int]):
        self._camera_handle.set(cv2.CAP_PROP_FRAME_WIDTH, value[0])
        self._camera_handle.set(cv2.CAP_PROP_FRAME_HEIGHT, value[1])

    @property
    def channels(self) -> int:
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value

    @property
    def refresh_rate(self) -> float:
        return self._camera_handle.get(cv2.CAP_PROP_FPS)

    @refresh_rate.setter
    def refresh_rate(self, value: float):
        self._camera_handle.set(cv2.CAP_PROP_FPS, value)

    @property
    def shape(self):
        w, h = self.resolution

        if self.channels == 1: return (h, w)
        
        return (h, w, self.channels)

    def capture(self) -> image.Image:
        ret, raw_data = self._camera_handle.read()

        if not ret:
            # Couldn't capture image, throw exception
            raise Exception("Could not capture an image with the CV2Camera")
        
        # Use float (spec of program)!
        raw_data = image.ToFloat(raw_data)

        # Undistort if can
        # rawImg = self.characterisation.Undistort(rawImg)

        # Convert to grey if needed
        if (self.channels == 1) and (1 < len(raw_data.shape)):
            raw_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)

        return image.Image(raw_data)

    def __del__(self):
        if self._camera_handle:
            if self._camera_handle.isOpened():
                self._camera_handle.release()

class FileCamera(BaseCamera):
    _exclude_fields = {
        '_images', 
        '_prefetch', '_preloaded_count', '_stop_event', '_prefetcher_thread', '_lock', '_xp'
    }

    def __init__(self, resolution: tuple[int, int], channels: int, refresh_rate: float, images:list[image.FileImage]=None, prefetch=-1, 
        char:ch.ZhangChar=None
    ):
        super().__init__(char=char)

        self._images = images
        self._resolution = resolution
        self._refresh_rate = refresh_rate
        self._channels = channels

        self._prefetch = prefetch
        self._preloaded_count = 0

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._prefetcher_thread = None

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution
    
    @resolution.setter
    def resolution(self, value: tuple[int, int]):
        self._resolution = value

    @property
    def channels(self) -> int:
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value

    @property
    def refresh_rate(self) -> float:
        return self._refresh_rate

    @refresh_rate.setter
    def refresh_rate(self, value: float):
        self._refresh_rate = value

    @property
    def shape(self):
        w, h = self.resolution

        if self.channels == 1: return (h, w)
        
        return (h, w, self.channels)

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
    def channels(self) -> int:
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
    @abstractmethod
    def aspect_ratio(self) -> float:
        raise NotImplementedError
    
    @property
    def shape(self):
        w, h = self.resolution

        if self.channels == 1:
            return (h, w)
        
        return (h, w, self.channels)

    @property
    def debug(self):
        return self._debug
    
    @debug.setter
    def debug(self, value):
        self._debug = value

    @abstractmethod
    def display(self, img):
        raise NotImplementedError
    
    def __str__(self):
        v = "<Projector>"

        if self.char:
            v += f" {self.char}"

        else: v+= " (Not Characterised)"

        return v


# Utility methods

def gather_gamma_imgs(camera: BaseCamera, projector: BaseProjector, intensities):
    xp = utils.ProcessingContext().xp

    captured_imgs = xp.empty((len(intensities), *camera.shape), dtype=xp.float32)

    for i, v in enumerate(intensities):
        project_img = xp.ones(projector.shape, dtype=xp.float32) * v

        # TODO: Callbacks
        projector.display(project_img)

        captured_imgs[i] = camera.capture().raw_data

    return captured_imgs

    # Find first observable change of values for averages (left and right sides) i.e >= delta
    # s, f = DetectableIndices(averages, delta)

    # vis_averages = averages[s:f+1]
    # vis_intensities = intensities[s:f+1]

    # coeffs = xp.polyfit(vis_averages, vis_intensities, order)
    # visible = intensities[s:f+1]

    # return coeffs, visible