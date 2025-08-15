import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from abc import ABC

from .devices.vision import VisionConfig

from .utils import ProcessingContext, AlwaysNumpy

class Image(ABC):
    def __init__(self, data):
        self.m_RawData = data

    @property
    def rawData(self):
        return self.m_RawData

class FileImage(Image):
    def __init__(self, path: Path, preload=False):
        super().__init__(None)

        self.m_Path: Path = path

        if preload: _ = self.rawData

    @property
    def rawData(self):
        # Check if the data needs to be loaded
        if self.m_RawData is None:
            self.m_RawData = cv2.imread(str(self.m_Path.resolve()), flags=cv2.IMREAD_UNCHANGED)
            self.m_RawData = self.m_RawData.astype(np.float32) / 255.0 # Default to float32

        return self.m_RawData

    def __str__(self):
        return f"{self.m_Path.absolute()}"

def Undistort(img_data, config: VisionConfig):
    return cv2.undistort(img_data, config.intrinsicMat, config.distortMat, None, config.intrinsicMat)  

def AddGaussianNoise(rawData, sigma=0.01, mean=0.0, clip=True):
    xp = ProcessingContext().xp

    rawData = xp.asarray(rawData)
    rawData += xp.random.normal(mean, sigma, size=rawData.shape)

    # Check if clip is enabled
    if clip: return Clip(rawData)

    return rawData

def AddSaltPepperNoise(rawData, saltPercent=0.05, pepperPercent=0.05):
    xp = ProcessingContext().xp

    randNoise = xp.random.rand(*rawData.shape)

    if 0.0 < saltPercent:
        saltMask = randNoise < saltPercent
        rawData[saltMask] = 1.0 if rawData.dtype == xp.float32 else 255

    if 0.0 < pepperPercent:
        pepperMask = (randNoise >= saltPercent) & (randNoise < saltPercent + pepperPercent)
        rawData[pepperMask] = 0.0 if rawData.dtype == xp.float32 else 255

    return rawData

def Clip(rawData):
    xp = ProcessingContext().xp
    
    if rawData.dtype == xp.uint8: rawData = xp.clip(rawData, 0, 255)
    else: rawData = xp.clip(rawData, 0.0, 1.0)

    return rawData

def ToGrey(rawData):
    xp = ProcessingContext().xp

    rawData = xp.asarray(rawData)

    if rawData.ndim == 2: return rawData
    
    if rawData.ndim == 3:
        h, w, c = rawData.shape
        if c == 1: return rawData.squeeze()
        if c == 3: return rawData[:, :, 2].squeeze() # Keep red channel for now

    raise Exception("Image is in unrecognised format")

def ToFloat(rawData):
    xp = ProcessingContext().xp

    rawData = xp.asarray(rawData)

    if rawData.dtype == xp.float32 or (rawData.dtype == xp.float64):
        return rawData

    if rawData.dtype == xp.uint8:
        return rawData.astype(xp.float32) / 255.0
    
    raise Exception(f"Image must be in integer format (found {rawData.dtype})")

def ExpandN(rawData, N=3):
    xp = ProcessingContext().xp

    rawData = xp.asarray(rawData)

    return xp.dstack([rawData] * N)

def ToU8(rawData):
    xp = ProcessingContext().xp

    if rawData.dtype == xp.uint8:
        return rawData

    if (rawData.dtype != xp.float32) and (rawData.dtype != xp.float64):
        raise Exception(f"Image must be in float format (found {rawData.dtype})")
    
    return (rawData * 255.0).astype(xp.uint8)

def ThresholdMask(rawData, threshold=0.1):
    xp = ProcessingContext().xp

    rawData = xp.asarray(rawData.copy())

    belowMask = rawData < threshold

    rawData[belowMask] = 0.0
    rawData[~belowMask] = 1.0
    
    return rawData

def DC(imgs):
    xp = ProcessingContext().xp
    
    """ Calculate average intensity across supplied imgs (return uint8 format)"""
    imgs = xp.asarray(imgs)

    return xp.sum(imgs, axis=0, dtype=xp.float32) / len(imgs)

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

    if size is None:
        h, w = rawData.shape[:2]
        rawData = cv2.resize(rawData, (int(1000 * (w / h)), 1000))
    
    cv2.imshow(name, rawData)
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