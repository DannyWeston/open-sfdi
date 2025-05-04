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

        self._path = path

        self.__greyscale = greyscale

    @property
    def raw_data(self) -> np.ndarray:
        # Check if the data needs to be loaded
        if self._raw_data is None:
            self._raw_data = cv2.imread(str(self._path.resolve()), cv2.IMREAD_COLOR)
            self._raw_data = self._raw_data.astype(np.float32) / 255.0

            if self.__greyscale: self._raw_data = to_grey(self._raw_data)

        # Default to float32
        return super().raw_data

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

    if img_data.dtype != int:
        raise Exception(f"Image must be in integer format (found {img_data.dtype})")

    return img_data.astype(np.float32) / 255.0

def to_int8(img_data) -> np.ndarray:
    if img_data.dtype == np.uint8:
        return img_data

    if img_data.dtype != np.float32:
        raise Exception(f"Image must be in float format (found {img_data.dtype})")
    
    return (img_data * 255.0).astype(np.uint8)

def find_corners(img, cb_size, window_size):
    cb_algo_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    # cb_algo_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_EXHAUSTIVE

    uint_img = to_int8(img)
    result, corners = cv2.findChessboardCorners(uint_img, cb_size, flags=cb_algo_flags)

    if not result: return None

    # Refine pixels identified to subpixels
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    corner_subpixels = cv2.cornerSubPix(uint_img, corners, window_size, (-1, -1), criteria)
    return np.squeeze(corner_subpixels)

def dc_imgs(imgs) -> np.ndarray:
    """ Calculate average intensity across supplied imgs (return uint8 format)"""
    return np.sum(np.array(imgs), axis=0) / len(imgs)

def calc_vignetting(img: np.ndarray, expected_max=None):
    if expected_max is None:
        expected_max = img.max()

    ideal_img = np.ones_like(img) * expected_max

    return ideal_img - img

def calc_gamma(img: np.ndarray):
    if img.ndim == 3:
        return np.mean(img)
        
    return np.mean(img)

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

    N = len(xss)
    colors = np.linspace(0.0, 1.0, N)

    for i in range(N):
        ax1.scatter(xss[i], yss[i], color=(colors[i], 0.0, 0.0))

    plt.show(block=True)