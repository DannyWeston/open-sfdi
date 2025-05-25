import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from abc import ABC

class Image(ABC):
    def __init__(self, data: np.ndarray):
        self._raw_data = data

    @property
    def raw_data(self) -> np.ndarray:
        return self._raw_data

# Images can be lazy loaded making use of lazy loading pattern
class FileImage(Image):
    def __init__(self, path: Path, greyscale=False):
        super().__init__(None)

        self._path: Path = path

        self.__greyscale = greyscale

    @property
    def raw_data(self) -> np.ndarray:
        # Check if the data needs to be loaded
        if self._raw_data is None:
            self._raw_data = cv2.imread(str(self._path.resolve()), cv2.IMREAD_COLOR)
            self._raw_data = self._raw_data.astype(np.float32) / 255.0 # Default to float32

        # Change to greyscale if needed
        if self.__greyscale: 
            self._raw_data = to_grey(self._raw_data)

        return self._raw_data

    def __str__(self):
        return f"{self._path.absolute()}"

def undistort_img(img_data, intrinsic_mat, dist_mat):
    return cv2.undistort(img_data, intrinsic_mat, dist_mat, None, intrinsic_mat)  

def to_grey(img_data: np.ndarray) -> np.ndarray:
    if img_data.ndim == 2: return img_data
    
    if img_data.ndim == 3:
        h, w, c = img_data.shape
        if c == 1: return img_data.squeeze()
        if c == 3: return cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)

    raise Exception("Image is in unrecognised format")

def to_f32(img_data) -> np.ndarray:
    if img_data.dtype == np.float32:
        return img_data

    if img_data.dtype == int or img_data.dtype == cv2.CV_8U or img_data.dtype == np.uint8:
        return img_data.astype(np.float32) / 255.0
    
    raise Exception(f"Image must be in integer format (found {img_data.dtype})")

def to_int8(img_data) -> np.ndarray:
    if img_data.dtype == np.uint8:
        return img_data

    if img_data.dtype != np.float32:
        raise Exception(f"Image must be in float format (found {img_data.dtype})")
    
    return (img_data * 255.0).astype(np.uint8)

def flip_colours(img_data):
    if img_data.ndim != 3:
        return img_data
    
    return cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

def find_corners(img, cb_size):
    uint_img = to_int8(img)

    # flags = cv2.CALIB_CB_EXHAUSTIVE
    # result, corners = cv2.findChessboardCornersSB(uint_img, cb_size, flags=flags)
    # if not result: return None

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_EXHAUSTIVE
    result, corners = cv2.findChessboardCorners(uint_img, cb_size, flags=flags)

    if not result:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(uint_img, corners, (15, 15), (-1, -1), criteria)

    return corners.squeeze()

def threshold_mask(img, threshold=0.004, max=1.0, type=cv2.THRESH_BINARY):
    success, result = cv2.threshold(img, threshold, max, type)

    if not success:
        return None
    
    return result

def calc_modulation(imgs, phases):
    N = len(phases)

    a = np.zeros_like(imgs[0])
    b = np.zeros_like(a)

    for i, phase in enumerate(phases):
        a += np.square(imgs[i] * np.sin(phase))
        b += np.square(imgs[i] * np.cos(phase))

    return (2.0 / N) * np.sqrt(a + b)

def dc_imgs(imgs) -> np.ndarray:
    """ Calculate average intensity across supplied imgs (return uint8 format)"""
    return np.sum(imgs, axis=0, dtype=np.float32) / len(imgs)

def calc_vignetting(img: np.ndarray, expected_max=None):
    if expected_max is None:
        expected_max = img.max()

    ideal_img = np.ones_like(img) * expected_max

    return ideal_img - img

def calc_gamma(img: np.ndarray):
    kernel = (9, 16) # 9 pixels tall, 16 wide

    h = int(img.shape[0] / 2)
    h1 = h - kernel[0]
    h2 = h + kernel[0]

    w = int(img.shape[1] / 2)
    w1 = w - kernel[1]
    w2 = w + kernel[1]

    roi = img[h1:h2, w1:w2]

    return np.mean(roi)

def show_image(img, name='Image', size=None, wait=0):
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