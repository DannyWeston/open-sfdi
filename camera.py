import cv2

class Camera:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_SETTINGS, 1)

    def show_raw_feed(self):
        while True:
            ret_val, img = self.camera.read()

            if ret_val:
                cv2.imshow(self.camera_id, img)

                if cv2.waitKey(1) == 27: break  # esc to quit

        cv2.destroyAllWindows()