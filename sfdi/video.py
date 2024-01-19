class Projector:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def display(self, img):
        pass

class Camera:
    def __init__(self, settings):
        self.settings = settings

    def capture(self):
        pass
    
class FileCamera(Camera):
    def __init__(self, img_paths):
        super().__init__(settings=None)
            
        self._image_num = 0
        
        self._images = []
        
        for img_path in img_paths:
            self._images.append(cv2.imread(img_path, 1))
        
    def capture(self):
        img = self._images[self._image]
        
        self._image_num = (self._image_num + 1) % len(self._images)
        
        return img
    
class CallbackCamera(Camera):
    def __init__(self, cb):
        super().__init__(settings=None)
        
        self._cb = cb
        
    def capture(self):
        return self._cb()
    
class FakeCamera(Camera):
    def __init__(self, imgs):
        super().__init__(settings=None)
        
        self._images = imgs
        self._image_num = 0
        
    def capture(self):
        img = self._images[self._image]
        
        self._image_num = (self._image_num + 1) % len(self._images)
        
        return img
    
    def add_image(self, img):
        self._images.append(img)