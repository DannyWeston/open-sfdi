from . import devices, image, characterisation as ch

def stereo_images(camera1: devices.BaseCamera, camera2: devices.BaseCamera, undistort=False) -> tuple[image.Image, image.Image]:
    # Ideally we want this to happen at the same time
    img1 = camera1.capture()
    img2 = camera2.capture()

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

class StereoMeasurement:
    def __init__(self, left_camera: devices.BaseCamera, right_camera: devices.BaseCamera):

        if left_camera.char is None: raise ch.NotCharacterisedException("The left camera is not characterised!")
        if right_camera.char is None: raise ch.NotCharacterisedException("The right camera is not characterised!")
        
        self._left_camera = left_camera
        self._right_camera = right_camera

        self._switched = False

    @property
    def left_camera(self):
        return self._left_camera

    @property
    def right_camera(self):
        return self._right_camera
    
    def switch_cameras(self):
        self.left_camera, self.right_camera = self.right_camera, self.left_camera

    def measure(self):
        raise NotImplementedError