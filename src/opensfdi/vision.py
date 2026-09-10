import cv2
import numpy as np

from . import image, characterisation as ch, utils
from .devices import camera
from .devices import projector

def stereo_images(camera1: camera.Camera, camera2: camera.Camera, undistort=False) -> tuple[image.Image, image.Image]:
    # Ideally we want this to happen at the same time
    img1 = camera1.load()
    img2 = camera2.load()

    # Undistort the images
    if undistort:
        img1 = image.Image(camera1.get_char().undistort_img(img1.raw_data))
        img2 = image.Image(camera2.get_char().undistort_img(img2.raw_data))

    return img1, img2

class StereoCharacterisation:
    def __init__(self, left_camera: camera.Camera, right_camera: camera.Camera, char_board: ch.CharacterisationBoard):
        self._left_camera = left_camera
        self._right_camera = right_camera

        self._char_board = char_board

        self._cameras_switched = False

        self._characterising = False

        self._on_capture_callbacks = []

    @property
    def cameras_switched(self) -> bool:
        return self._cameras_switched
    
    @cameras_switched.setter
    def cameras_switched(self, value: bool):
        self._cameras_switched = value

    @property
    def characterising(self) -> bool:
        return self._characterising
    
    @characterising.setter
    def characterising(self, value: bool):
        self._characterising = value

    def add_capture_callback(self, func):
        self._on_capture_callbacks.append(func)

class StereoMeasurementSettings:
    def __init__(self):
        pass

class StereoMeasurement:
    def __init__(self, left_camera: camera.Camera, right_camera: camera.Camera):

        if left_camera.char is None: raise ch.NotCharacterisedException("The left camera is not characterised!")
        if right_camera.char is None: raise ch.NotCharacterisedException("The right camera is not characterised!")
        
        self._left_camera = left_camera
        self._right_camera = right_camera

        self._switched = False
    
    def switch_cameras(self):
        self.left_camera, self.right_camera = self.right_camera, self.left_camera

    def measure(self, left_img, right_img):
        # Convert to grayscale for stereo matching (color can be used for point cloud coloring)
        left_grey_img = image.ToGrey(self._left_img.raw_data)
        right_grey_img =  image.ToGrey(self._right_img.raw_data)

        left_rec = cv2.remap(left_grey_img, self.left_map_1, self.left_map_2, cv2.INTER_LINEAR)
        right_rec = cv2.remap(right_grey_img, self.right_map_1, self.right_map_2, cv2.INTER_LINEAR)

        # Also rectify color images for colored point cloud
        # left_rec_rgb = cv2.remap(self._left_img.raw_data, left_map_1, left_map_2, cv2.INTER_LINEAR)

        self._disparity = self._stereo_matcher.compute(left_rec, right_rec)
        if self._disparity is None: return False
        
        self._disparity = self._disparity.astype(np.float32) / 16.0

        self._disparity = self._disparity.copy()
        self._disparity[self._disparity < 0] = 0
        debug_max_disparity = self._disparity.max()

        if debug_max_disparity > 0: self._disparity = (self._disparity / debug_max_disparity) * 255.0

        self._disparity = self._disparity.astype(np.uint8)

        self.disparity.emit(image.Image(self._disparity))

        self._left_img = None
        self._right_img = None

        # TODO: Create a mask for valid points in disparity map
        # disparityMask = disparity > disparity.min()

        return True