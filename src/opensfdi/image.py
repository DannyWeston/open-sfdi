import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from abc import ABC

from .devices.vision import VisionConfig

class Image(ABC):
    def __init__(self, data: np.ndarray):
        self._raw_data = data

    @property
    def raw_data(self) -> np.ndarray:
        return self._raw_data

# Images can be lazy loaded making use of lazy loading pattern
class FileImage(Image):
    def __init__(self, path: Path, channels=1):
        super().__init__(None)

        self._path: Path = path

        self.__channels = channels

    @property
    def raw_data(self) -> np.ndarray:
        # Check if the data needs to be loaded
        if self._raw_data is None:
            flags = None
            if self.__channels == 3:
                flags = cv2.IMREAD_COLOR
            elif self.__channels == 1:
                flags = cv2.IMREAD_GRAYSCALE

            self._raw_data = cv2.imread(str(self._path.resolve()), flags)
            self._raw_data = self._raw_data.astype(np.float32) / 255.0 # Default to float32

        return self._raw_data

    def __str__(self):
        return f"{self._path.absolute()}"

def Undistort(img_data, config: VisionConfig):
    return cv2.undistort(img_data, config.intrinsicMat, config.distortMat, None, config.intrinsicMat)  

def AddGaussian(img_data, sigma=0.01, mean=0.0, clip=True):
    img_data = img_data + np.random.normal(mean, sigma, size=img_data.shape)

    if clip: img_data = np.clip(img_data, 0.0, 1.0, dtype=np.float32) 

    return img_data

def ToGrey(img_data: np.ndarray) -> np.ndarray:
    if img_data.ndim == 2: return img_data
    
    if img_data.ndim == 3:
        h, w, c = img_data.shape
        if c == 1: return img_data.squeeze()
        if c == 3: return cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)

    raise Exception("Image is in unrecognised format")

def ToF32(img_data) -> np.ndarray:
    if img_data.dtype == np.float32:
        return img_data

    if img_data.dtype == int or img_data.dtype == cv2.CV_8U or img_data.dtype == np.uint8:
        return img_data.astype(np.float32) / 255.0
    
    raise Exception(f"Image must be in integer format (found {img_data.dtype})")

def ToU8(img_data) -> np.ndarray:
    if img_data.dtype == np.uint8:
        return img_data

    if img_data.dtype != np.float32:
        raise Exception(f"Image must be in float format (found {img_data.dtype})")
    
    return (img_data * 255.0).astype(np.uint8)

def ThresholdMask(img, threshold=0.004, max=1.0, type=cv2.THRESH_BINARY):
    success, result = cv2.threshold(img, threshold, max, type)

    if not success: return None
    
    return result

def CalculateModulation(imgs, phases):
    N = len(phases)

    a = np.zeros_like(imgs[0])
    b = np.zeros_like(a)

    for i, phase in enumerate(phases):
        a += np.square(imgs[i] * np.sin(phase))
        b += np.square(imgs[i] * np.cos(phase))

    return (2.0 / N) * np.sqrt(a + b)

def DC(imgs) -> np.ndarray:
    """ Calculate average intensity across supplied imgs (return uint8 format)"""
    return np.sum(imgs, axis=0, dtype=np.float32) / len(imgs)

def CalculateVignette(img: np.ndarray, expected_max=None):
    if expected_max is None:
        expected_max = img.max()

    ideal_img = np.ones_like(img) * expected_max

    return ideal_img - img

def CalculateGamma(img: np.ndarray):
    kernel = (9, 16) # 9 pixels tall, 16 wide

    h = int(img.shape[0] / 2)
    h1 = h - kernel[0]
    h2 = h + kernel[0]

    w = int(img.shape[1] / 2)
    w1 = w - kernel[1]
    w2 = w + kernel[1]

    roi = img[h1:h2, w1:w2]

    return np.mean(roi)

def Show(img: np.ndarray, name='Image', size=None, wait=0):
    if size is None: size = img.shape[1::-1]

    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 0:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    
    cv2.imshow(name, cv2.resize(img, size))
    cv2.resizeWindow(name, size[0], size[1])
    cv2.waitKey(wait)

def show_scatter(xss, yss):
    fig = plt.figure()
    ax1 = fig.add_subplot()

    ax1.set_xlim(0.0, 1.01)
    ax1.set_ylim(0.0, 1.01)

    N = len(xss)
    colors = np.linspace(0.0, 1.0, N)

    for i in range(N):
        ax1.scatter(xss[i], yss[i], color=(colors[i], 0.0, 0.0))

    plt.show(block=True)