import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from abc import ABC

from . import utils

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
        xp = utils.ProcessingContext().xp

        # Check if the data needs to be loaded
        if self.m_RawData is None:
            self.m_RawData = cv2.imread(str(self.m_Path.resolve()), flags=cv2.IMREAD_UNCHANGED)
            self.m_RawData = xp.asarray(self.m_RawData) / xp.iinfo(self.m_RawData.dtype).max # Default to float64

        return self.m_RawData

    def __str__(self):
        return f"{self.m_Path.absolute()}"

def AddGaussianNoise(rawData, sigma=0.01, mean=0.0, clip=True):
    xp = utils.ProcessingContext().xp

    rawData = xp.asarray(rawData)
    rawData += xp.random.normal(mean, sigma, size=rawData.shape)

    # Check if clip is enabled
    if clip: return Clip(rawData)

    return rawData

def AddSaltPepperNoise(rawData, saltPercent=0.05, pepperPercent=0.05):
    xp = utils.ProcessingContext().xp

    dtype = rawData.dtype
    if (dtype != xp.float32) or (dtype != xp.float64):
        raise Exception("rawData must be float-based format") 

    randNoise = xp.random.rand(*rawData.shape)

    if 0.0 < saltPercent:
        saltMask = randNoise < saltPercent
        rawData[saltMask] = 1.0

    if 0.0 < pepperPercent:
        pepperMask = (randNoise >= saltPercent) & (randNoise < saltPercent + pepperPercent)
        rawData[pepperMask] = 0.0

    return rawData

def Clip(rawData):
    xp = utils.ProcessingContext().xp
    
    if rawData.dtype == xp.uint8: rawData = xp.clip(rawData, 0, 255)
    else: rawData = xp.clip(rawData, 0.0, 1.0)

    return rawData

def ToGrey(rawData):
    xp = utils.ProcessingContext().xp

    rawData = xp.asarray(rawData)

    if rawData.ndim == 2: return rawData
    
    if rawData.ndim == 3:
        h, w, c = rawData.shape
        if c == 1: return rawData.squeeze()
        if c == 3: return rawData[:, :, 2].squeeze() # Keep red channel for now

    raise Exception("Image is in unrecognised format")

def ToFloat(rawData):
    xp = utils.ProcessingContext().xp

    rawData = xp.asarray(rawData)

    dtype = rawData.dtype

    if dtype == xp.float32 or dtype == xp.float64:
        return rawData

    if dtype == xp.uint8:
        return rawData.astype(xp.float32) / xp.iinfo(dtype).max

    if dtype == xp.uint16:
        return rawData.astype(xp.float64) / xp.iinfo(dtype).max
    
    raise Exception(f"Image must be in integer format (found {dtype})")

def ExpandN(rawData, N=3):
    xp = utils.ProcessingContext().xp

    rawData = xp.asarray(rawData)

    return xp.dstack([rawData] * N)

def ToInt(rawData):
    xp = utils.ProcessingContext().xp

    dtype = rawData.dtype

    if dtype == xp.uint8 or dtype == xp.uint16:
        return rawData
    
    if dtype == xp.float32:
        return (rawData * xp.iinfo(xp.uint8).max).astype(xp.uint8)

    if dtype == xp.float64:
        return (rawData * xp.iinfo(xp.uint16).max).astype(xp.uint16)
    
    raise Exception(f"Image must be in float format (found {rawData.dtype})")
    
def ThresholdMask(rawData, min=0.1, max=0.9, dtype=None):
    xp = utils.ProcessingContext().xp

    if dtype is None: dtype = xp.float32

    mask = xp.ones_like(rawData, dtype=dtype)

    low = rawData <= min
    mask[low] = 0.0

    high = rawData >= max
    mask[high] = 0.0
    
    return mask

def Normalise(data):
    xp = utils.ProcessingContext().xp

    a = xp.nanmin(data)
    b = xp.nanmax(data)
    data = (data - a) / (b - a)

    return data

def DC(imgs):
    xp = utils.ProcessingContext().xp
    
    """ Calculate average intensity across supplied imgs (return uint8 format)"""
    imgs = xp.asarray(imgs)

    return xp.sum(imgs, axis=0) / len(imgs)

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
    rawData = utils.ToNumpy(rawData)

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    if size is None: 
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        _, _, w, h = cv2.getWindowImageRect(name)

        rawData = cv2.resize(rawData, (w, h))

    else: 
        w, h = size
        rawData = cv2.resize(rawData, size)
        cv2.resizeWindow(name, w, h)

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