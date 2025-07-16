import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from abc import ABC

from .devices.vision import VisionConfig

from .utils import ProcessingContext, AlwaysNumpy

class Image(ABC):
    def __init__(self, data: np.ndarray):
        self.m_RawData = data

    @property
    def rawData(self) -> np.ndarray:
        return self.m_RawData

# Images can be lazy loaded making use of lazy loading pattern
class FileImage(Image):
    def __init__(self, path: Path):
        super().__init__(None)

        self.m_Path: Path = path

    @property
    def rawData(self) -> np.ndarray:
        # Check if the data needs to be loaded
        if self.m_RawData is None:
            self.m_RawData = cv2.imread(str(self.m_Path.resolve()), flags=cv2.IMREAD_UNCHANGED)
            self.m_RawData = self.m_RawData.astype(np.float32) / 255.0 # Default to float32

        return self.m_RawData

    def __str__(self):
        return f"{self.m_Path.absolute()}"

def Undistort(img_data, config: VisionConfig):
    return cv2.undistort(img_data, config.intrinsicMat, config.distortMat, None, config.intrinsicMat)  

def AddGaussian(rawData, sigma=0.01, mean=0.0, clip=True):
    rawData = rawData + np.random.normal(mean, sigma, size=rawData.shape)

    if clip: rawData = np.clip(rawData, 0.0, 1.0, dtype=np.float32) 

    return rawData

def ToGrey(rawData: np.ndarray) -> np.ndarray:
    if rawData.ndim == 2: return rawData
    
    if rawData.ndim == 3:
        h, w, c = rawData.shape
        if c == 1: return rawData.squeeze()
        if c == 3: return cv2.cvtColor(rawData, cv2.COLOR_BGR2GRAY)

    raise Exception("Image is in unrecognised format")

def ToF32(rawData) -> np.ndarray:
    if rawData.dtype == np.float32:
        return rawData

    if rawData.dtype == int or rawData.dtype == cv2.CV_8U or rawData.dtype == np.uint8:
        return rawData.astype(np.float32) / 255.0
    
    raise Exception(f"Image must be in integer format (found {rawData.dtype})")

def ToU8(rawData) -> np.ndarray:
    if rawData.dtype == np.uint8:
        return rawData

    if rawData.dtype != np.float32:
        raise Exception(f"Image must be in float format (found {rawData.dtype})")
    
    return (rawData * 255.0).astype(np.uint8)

def ThresholdMask(rawData, threshold=0.004, max=1.0, type=cv2.THRESH_BINARY):
    success, result = cv2.threshold(AlwaysNumpy(rawData), threshold, max, type)

    if not success: return None
    
    return result

def DC(imgs):
    xp = ProcessingContext().xp
    
    """ Calculate average intensity across supplied imgs (return uint8 format)"""
    imgs = xp.asarray(imgs)

    return xp.sum(imgs, axis=0, dtype=np.float32) / len(imgs)

def CalculateVignette(rawData: np.ndarray, expectedMax=None):
    if expectedMax is None: expectedMax = rawData.max()

    idealImg = np.ones_like(rawData) * expectedMax

    return idealImg - rawData

def CalculateGamma(rawData: np.ndarray, kernel=(9, 16)):
    h = int(rawData.shape[0] / 2)
    h1 = h - kernel[0]
    h2 = h + kernel[0]

    w = int(rawData.shape[1] / 2)
    w1 = w - kernel[1]
    w2 = w + kernel[1]

    roi = rawData[h1:h2, w1:w2]

    return np.mean(roi)

def Show(rawData, name='Image', wait=0, size=None):
    # cv2 needs numpy
    rawData = AlwaysNumpy(rawData)

    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 0:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    
    cv2.imshow(name, rawData if size is None else cv2.resize(rawData, size))
    cv2.waitKey(wait)

def ShowScatter(xss, yss):
    fig = plt.figure()
    ax1 = fig.add_subplot()

    ax1.set_xlim(0.0, 1.01)
    ax1.set_ylim(0.0, 1.01)

    N = len(xss)
    colors = np.linspace(0.0, 1.0, N)

    for i in range(N):
        ax1.scatter(xss[i], yss[i], color=(colors[i], 0.0, 0.0))

    plt.show(block=True)