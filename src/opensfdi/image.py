import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from abc import ABC

from . import utils

# Image classes

class Image(ABC):
    def __init__(self, data):
        self.m_RawData = data

    @property
    def raw_data(self):
        return self.m_RawData

class FileImage(Image):
    def __init__(self, path: Path):
        super().__init__(None)

        self._path: Path = path

        self._preloaded = False

    @property
    def raw_data(self):
        # Check if the data needs to be loaded
        if not self._preloaded:
            self.Preload()

        return super().raw_data
    
    @property
    def path(self) -> Path:
        return self._path

    def Preload(self):
        if not self._preloaded:
            # cv2 load
            self.m_RawData = cv2.imread(str(self._path.resolve()), flags=cv2.IMREAD_UNCHANGED)
            self.m_RawData = self.m_RawData.astype(np.float32) / np.iinfo(self.m_RawData.dtype).max # Default to float32
            
            self._preloaded = True

    def __str__(self):
        return f"<FileImage> : {self.path.absolute()}"


# Utility methods

def ToFloat(rawData):
    xp = utils.ProcessingContext().xp

    rawData = xp.asarray(rawData)

    dtype = rawData.dtype

    if dtype == xp.float32:
        return rawData

    if (dtype == xp.uint8) or (dtype == xp.int16):
        return rawData.astype(xp.float32) / xp.iinfo(dtype).max

    raise Exception(f"Image must be in integer format (found {dtype})")

def ToInt(rawData):
    xp = utils.ProcessingContext().xp

    dtype = rawData.dtype

    if (dtype == xp.uint8) or (dtype == xp.uint16):
        return rawData
    
    if (dtype == xp.float32) or (dtype == xp.float64):
        return (rawData * xp.iinfo(xp.uint8).max).astype(xp.uint8)
    
    raise Exception(f"Image must be in float format (found {rawData.dtype})")

def ToGrey(rawData):
    if rawData.ndim == 2: 
        return rawData
    
    if rawData.ndim == 3:
        c = rawData.shape[2]
        if c == 1: return rawData.squeeze()
        if c == 3: return rawData[:, :, 2].squeeze() # Keep red channel for now

    raise Exception("Image is in unrecognised format")

def Clip(rawData):
    xp = utils.ProcessingContext().xp
    
    if rawData.dtype == xp.uint8: rawData = xp.clip(rawData, 0, 255)
    else: rawData = xp.clip(rawData, 0.0, 1.0)

    return rawData

def CoordBilinearInterp(img, coords):
    xp = utils.ProcessingContext().xp
    whole = coords.astype(xp.uint16)
    frac = coords - whole

    x1 = whole[0]
    x2 = x1 + 1

    y1 = whole[1]
    y2 = y1 + 1

    # a1 <-> a2
    # ^      ^
    # |      |
    # v      v
    # a3 <-> a4
    a1 = img[y1, x1]
    a2 = img[y1, x2]
    a3 = img[y2, x1]
    a4 = img[y2, x2]

    # x-direction
    r0 = (a1 * (1 - frac[0])) + (a2 * frac[0])
    r1 = (a3 * (1 - frac[0])) + (a4 * frac[0])

    # y-direction
    v = (r0 * (1 - frac[1])) + (r1 * frac[1])

    return v

def ExpandN(rawData, N=3):
    xp = utils.ProcessingContext().xp

    rawData = xp.asarray(rawData)

    return xp.dstack([rawData] * N)

def ThresholdMask(data, min=0.1, max=0.9):
    xp = utils.ProcessingContext().xp

    mask = xp.ones(data.shape, dtype=xp.bool_)

    valid = (data > min) & (data < max)
    
    return mask & valid

def Normalise(data):
    xp = utils.ProcessingContext().xp

    a = xp.nanmin(data)
    b = xp.nanmax(data)
    data = (data - a) / (b - a)

    return data

# def CalculateVignette(rawData: np.ndarray, expectedMax=None):
#     if expectedMax is None: expectedMax = rawData.max()

#     idealImg = np.ones_like(rawData) * expectedMax

#     return idealImg - rawData


# Fringe Projection

def make_fringe_pattern(resolution, num_stripes, phase=0.0, rotation=0.0, channels=1, dst=None):
    ''' 
        resolution: (width, height) in integer pixels\n
        num_stripes: float for total number of oscillations\n
        phase: float in radians for signal phase shift\n
        rotation: float in radians for orientation of fringes\n
    '''

    xp = utils.ProcessingContext().xp

    w, h = resolution

    ys, xs = xp.meshgrid(
        xp.linspace(0.0, 1.0, h, endpoint=False, dtype=xp.float32),
        xp.linspace(0.0, 1.0, w, endpoint=False, dtype=xp.float32),
        indexing='ij'
    )

    pixels = (xs * xp.cos(rotation)) - (ys * xp.sin(rotation))

    # I(x, y) = cos(2 * pi * f * x - phi)
    fringes = xp.cos(num_stripes * 2.0 * xp.pi * pixels - phase, dtype=xp.float32)

    # Normalise fringes from [-1..1] to [0..1]
    fringes += 0.5
    fringes /= 2.0

    if channels == 1: return fringes

    return xp.dstack([fringes] * channels)

def ac_component(imgs):
    (2.0 / N) * xp.sqrt(a ** 2 + b ** 2)

# Noise

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
    if dtype != xp.float32:
        raise Exception("rawData must be float-based format") 

    randNoise = xp.random.rand(*rawData.shape)

    if 0.0 < saltPercent:
        saltMask = randNoise < saltPercent
        rawData[saltMask] = 1.0

    if 0.0 < pepperPercent:
        pepperMask = (randNoise >= saltPercent) & (randNoise < saltPercent + pepperPercent)
        rawData[pepperMask] = 0.0

    return rawData


# Preview Methods

def show_img(rawData, name='Image', wait=0, size=None):
    with utils.ProcessingContext.UseGPU(False):     # cv2 needs numpy
        xp = utils.ProcessingContext().xp

        rawData = utils.ToContext(xp, rawData)

        h, w, *_ = rawData.shape

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        if size:
            if size == "fullscreen": 
                cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
                _, _, w, h = cv2.getWindowImageRect(name)

            else: w, h = size

        rawData = cv2.resize(rawData, (w, h))
        cv2.resizeWindow(name, w, h)

        cv2.imshow(name, rawData)

        return cv2.waitKey(wait)

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