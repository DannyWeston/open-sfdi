import cv2

from typing import Protocol, Callable, Any
from dataclasses import dataclass

from . import devices, image, characterisation as ch, utils

def stereo_images(camera1: devices.BaseCamera, camera2: devices.BaseCamera, undistort=False) -> tuple[image.Image, image.Image]:
    # Ideally we want this to happen at the same time
    img1 = camera1.read()
    img2 = camera2.read()

    # Undistort the images
    if undistort:
        img1 = image.Image(camera1.char.undistort_img(img1.raw_data))
        img2 = image.Image(camera2.char.undistort_img(img2.raw_data))

    return img1, img2

class StereoCharacterisation:
    def __init__(self, left_camera: devices.BaseCamera, right_camera: devices.BaseCamera, char_board: ch.CharacterisationBoard):
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

class StereoMeasureModel:
    leftCameraCaptured = Signal(image.Image)
    rightCameraCaptured = Signal(image.Image)

    disparity = Signal(image.Image)
    pointcloud = Signal(Points)

    capturedListUpdated = Signal(int)

    def __init__(self, left_camera: devices.BaseCamera, right_camera: devices.BaseCamera):
        super().__init__()

        self.on_capture_list_updated = utils.EventEmitter()
        self.on_left_camera_captured = utils.EventEmitter()
        self.on_right_camera_captured = utils.EventEmitter()
        self.on_disparity = utils.EventEmitter()
        self.on_pointcloud = utils.EventEmitter()

        self.left_camera = left_camera
        self.right_camera = right_camera

        self._left_worker = CameraCaptureWorker(left_camera.camera)
        self._left_worker.captured.connect(self._left_captured)

        self._right_worker = CameraCaptureWorker(right_camera.camera)
        self._right_worker.captured.connect(self._right_captured)

        # Cache for data
        self._save_name = None
        self._i = 0

        self._img_name = "measurement"

        self._left_img = None
        self._right_img = None

        self._left_img_timestamps: list = None
        self._right_img_timestamps: list = None

        self._init_rectification()
        self._init_matcher()

    @property
    def img_repo(self):
        return self._img_repo

    @property
    def flags(self):
        flags = 0

        # Add internal flags

        return 0

    def start_session(self):
        self._left_img_timestamps = []
        self._right_img_timestamps = []
        self._disparity = None

        self._left_worker.start()
        self._right_worker.start()
   
    def revert_capture(self, save_imgs: bool):
        if len(self._left_img_timestamps) < 1: return False
        if len(self._right_img_timestamps) < 1: return False
        
        left_identifier = self._left_img_timestamps.pop()
        right_identifier = self._right_img_timestamps.pop()

        # Delete images on disk
        if save_imgs:
            self.img_repo.Delete(left_identifier)
            self.img_repo.Delete(right_identifier)

        return True

    def finish_session(self):
        pass

    def _init_rectification(self):
        left_camera = self.left_camera.camera
        right_camera = self.right_camera.camera
        
        # Compute rectification transforms
        c1 = left_camera.char
        c2 = right_camera.char

        R1, R2, P1, P2, self._q, roi1, roi2 = cv2.stereoRectify(
            c1.intrinsic_mat, c1.distort_mat, c2.intrinsic_mat, c2.distort_mat, 
            left_camera.resolution, c1.rotation_to_other, c1.translation_to_other,
            alpha=0
        )

        # Compute rectification maps
        self.left_map_1, self.left_map_2 = cv2.initUndistortRectifyMap(
            c1.intrinsic_mat, c1.distort_mat, R1, P1, left_camera.resolution, cv2.CV_32FC1
        )

        self.right_map_1, self.right_map_2 = cv2.initUndistortRectifyMap(
            c2.intrinsic_mat, c2.distort_mat, R2, P2, right_camera.resolution, cv2.CV_32FC1
        )

    def _init_matcher(self):
        # Create Stereo SGBM matcher - TUNE THESE PARAMETERS FOR YOUR SCENE!
        self._stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,        # Minimum disparity (usually 0)
            numDisparities=80,     # Search range: must be divisible by 16. Increase for closer objects.
            blockSize=5,           # Matching block size. Odd number between 3-11.
            P1=8 * 3 * 5**2,       # Smoothness penalty 1
            P2=32 * 3 * 5**2,      # Smoothness penalty 2 (usually 4*P1)
            disp12MaxDiff=1,       # Maximum allowed difference in left-right check
            uniquenessRatio=15,    # Margin in percent for uniqueness check
            speckleWindowSize=100, # Maximum size of smooth disparity regions
            speckleRange=2         # Maximum disparity variation within speckle window
        )

    def measure(self, save_imgs):
        if (self._left_img is None) or (self._right_img is None): return False

        now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e3)
        left_identifier = f"{self._img_name}_left{len(self._left_img_timestamps)}_{now}"
        right_identifier = f"{self._img_name}_right{len(self._right_img_timestamps)}_{now}"

        self._left_img_timestamps.append(left_identifier)
        self._right_img_timestamps.append(right_identifier)

        # TODO: Move to another thread to stop I/O spike?
        if save_imgs:
            self._img_repo.Add(self._left_img, left_identifier)
            self._img_repo.Add(self._right_img, right_identifier)

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

    def create_cloud(self):
        disparityMask = self._disparity > self._disparity.min()
        projected = cv2.reprojectImageTo3D(self._disparity, self._q)

        projected = projected[disparityMask]

        x = projected[:, 0]
        y = projected[:, 1]
        z = projected[:, 2]        

        # projected = projected[mask]

        points = Points(projected)
        points.lighting('off')
        points.cmap('viridis', projected[:, 2], name='DepthMap')

        self.pointcloud.emit(points)
    
    def cleanup(self):
        self._left_worker.stop()
        self._right_worker.stop()

    def _left_captured(self, img: image.Image):
        self._left_img = img

        self.leftCameraCaptured.emit(self._left_img)

    def _right_captured(self, img: image.Image):
        self._right_img = img

        self.rightCameraCaptured.emit(self._right_img)

class StereoMeasurement:
    def __init__(self, left_camera: devices.BaseCamera, right_camera: devices.BaseCamera):

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