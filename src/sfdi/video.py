import cv2

import subprocess
import io
import numpy as np

import pygame
import cv2

from picamera2 import Picamera2
from fractions import Fraction
from numpy.lib.stride_tricks import as_strided

DEFAULT_CAM_SETTINGS = {    
    "serial": '3816',
    "version" : 'v2.1 NoIR',
    "lett" : 'D_2f',
    "redgreen_gain" : Fraction(1 / 1),
    "bluegreen_gain" : Fraction(1 / 1),
    "resolution": (2592, 1936),
    "frame_rate": 1,
    "exposure": 'off',
    "shutter_speed": 10
}

class Projector:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def display(self, img):
        pass

class PygameProjector(Projector):
    def __init__(self, width, height):
        super().__init__(width, height)

        pygame.init()
        pygame.display.init()

        d_info = pygame.display.Info()
        self.d_width = d_info.current_w
        self.d_height = d_info.current_h

    def __del__(self):
        pygame.quit()

    def display(self, img):
        screen = pygame.display.set_mode((self.d_width, self.d_height), pygame.FULLSCREEN)
        screen.blit(pygame.transform.scale(img, (self.d_width, self.d_height)), (0, 0))
        pygame.display.flip()

class Camera:
    def __init__(self, settings = DEFAULT_CAM_SETTINGS):
        self.settings = settings

        self.width, self.height = settings["resolution"]

        cmd = 'cat /proc/cpuinfo | grep Serial | cut -d " " -f 2'
        child = subprocess.Popen(cmd, stdout = subprocess.PIPE, shell = True)
        output = child.communicate()[0]

        self.serial_number = output.decode("utf-8").strip("\n")

        Picamera2.set_logging(Picamera2.ERROR) # Remove console spam on init
        self.camera = Picamera2()
        camera_config = self.camera.create_still_configuration(
            main={"size": (1920, 1080)}, 
            lores={"size": (1920, 1080)},
            display="lores"
        )

        self.camera.configure(camera_config)
        self.camera.start()

        # self.camera.exposure_mode = settings["exposure"]    # Disable exposure
        # self.camera.shutter_speed = settings["shutter_speed"] # Shutter speed

        # self.camera.awb_mode = 'off'
        # self.camera.awb_gains = (settings["redgreen_gain"], settings["bluegreen_gain"]) # Set r/g & b/g gain

    def capture(self):
        return self.camera.capture_array("main")

    def __capture(self):
        stream = io.BytesIO()
        self.camera.capture(stream, format='jpeg', bayer=True)

        # Extract the raw Bayer data from the end of the stream, check the
        # header and strip it off before converting the data into a numpy array
        
        ver = {
            'RP_ov5647': 1,
            'RP_imx219': 2,
            }[self.camera.exif_tags['IFD0.Model']]

        offset = {
            1: 6404096,
            2: 10270208,
        }[ver]

        data = stream.getvalue()[-offset:]
        assert data[:4] == b'BRCM' # Remove
        data = data[32768:]
        data = np.fromstring(data, dtype=np.uint8)

        # For the V1 module, the data consists of 1952 rows of 3264 bytes of data.
        # The last 8 rows of data are unused (they only exist because the maximum
        # resolution of 1944 rows is rounded up to the nearest 16).
        #
        # For the V2 module, the data consists of 2480 rows of 4128 bytes of data.
        # There's actually 2464 rows of data, but the sensor's raw size is 2466
        # rows, rounded up to the nearest multiple of 16: 2480.
        #
        # Likewise, the last few bytes of each row are unused (why?). Here we
        # reshape the data and strip off the unused bytes.

        reshape, crop = {
            1: ((1952, 3264), (1944, 3240)),
            2: ((2480, 4128), (2464, 4100)),
            }[ver]
        data = data.reshape(reshape)[:crop[0], :crop[1]]

        # Horizontally, each row consists of 10-bit values. Every four bytes are
        # the high 8-bits of four values, and the 5th byte contains the packed low
        # 2-bits of the preceding four values. In other words, the bits of the
        # values A, B, C, D and arranged like so:
        #
        #  byte 1   byte 2   byte 3   byte 4   byte 5
        # AAAAAAAA BBBBBBBB CCCCCCCC DDDDDDDD AABBCCDD
        #
        # Here, we convert our data into a 16-bit array, shift all values left by
        # 2-bits and unpack the low-order bits from every 5th byte in each row,
        # then remove the columns containing the packed bits

        data = data.astype(np.uint16) << 2
        for byte in range(4):
            data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
        data = np.delete(data, np.s_[4::5], 1)

        # Now to split the data up into its red, green, and blue components. The
        # Bayer pattern of the OV5647 sensor is BGGR. In other words the first
        # row contains alternating green/blue elements, the second row contains
        # alternating red/green elements, and so on as illustrated below:
        #
        # GBGBGBGBGBGBGB
        # RGRGRGRGRGRGRG
        # GBGBGBGBGBGBGB
        # RGRGRGRGRGRGRG
        #
        # Please note that if you use vflip or hflip to change the orientation
        # of the capture, you must flip the Bayer pattern accordingly

        rgb = np.zeros(data.shape + (3, ), dtype=data.dtype)
        rgb[1::2, 0::2, 0] = data[1::2, 0::2] # Red
        rgb[0::2, 0::2, 1] = data[0::2, 0::2] # Green
        rgb[1::2, 1::2, 1] = data[1::2, 1::2] # Green
        rgb[0::2, 1::2, 2] = data[0::2, 1::2] # Blue

        # At this point we now have the raw Bayer data with the correct values
        # and colors but the data still requires de-mosaicing and
        # post-processing. If you wish to do this yourself, end the script here!
        #
        # Below we present a fairly naive de-mosaic method that simply
        # calculates the weighted average of a pixel based on the pixels
        # surrounding it. The weighting is provided by a byte representation of
        # the Bayer filter which we construct first:

        bayer = np.zeros(rgb.shape, dtype=np.uint8)
        bayer[1::2, 0::2, 0] = 1 # Red
        bayer[0::2, 0::2, 1] = 1 # Green
        bayer[1::2, 1::2, 1] = 1 # Green
        bayer[0::2, 1::2, 2] = 1 # Blue

        # Allocate an array to hold our output with the same shape as the input
        # data. After this we define the size of window that will be used to
        # calculate each weighted average (3x3). Then we pad out the rgb and
        # bayer arrays, adding blank pixels at their edges to compensate for the
        # size of the window when calculating averages for edge pixels.

        output = np.empty(rgb.shape, dtype=rgb.dtype)
        window = (3, 3)
        borders = (window[0] - 1, window[1] - 1)
        border = (borders[0] // 2, borders[1] // 2)

        rgb_pad = np.zeros((
            rgb.shape[0] + borders[0],
            rgb.shape[1] + borders[1],
            rgb.shape[2]), dtype=rgb.dtype)
        rgb_pad[
            border[0]:rgb_pad.shape[0] - border[0],
            border[1]:rgb_pad.shape[1] - border[1],
            :] = rgb
        rgb = rgb_pad

        bayer_pad = np.zeros((
            bayer.shape[0] + borders[0],
            bayer.shape[1] + borders[1],
            bayer.shape[2]), dtype=bayer.dtype)
        bayer_pad[
            border[0]:bayer_pad.shape[0] - border[0],
            border[1]:bayer_pad.shape[1] - border[1],
            :] = bayer
        bayer = bayer_pad

        # In numpy >=1.7.0 just use np.pad (version in Raspbian is 1.6.2 at the
        # time of writing...)
        #
        #rgb = np.pad(rgb, [
        #    (border[0], border[0]),
        #    (border[1], border[1]),
        #    (0, 0),
        #    ], 'constant')
        #bayer = np.pad(bayer, [
        #    (border[0], border[0]),
        #    (border[1], border[1]),
        #    (0, 0),
        #    ], 'constant')

        # For each plane in the RGB data, we use a nifty numpy trick
        # (as_strided) to construct a view over the plane of 3x3 matrices. We do
        # the same for the bayer array, then use Einstein summation on each
        # (np.sum is simpler, but copies the data so it's slower), and divide
        # the results to get our weighted average:

        for plane in range(3):
            p = rgb[..., plane]
            b = bayer[..., plane]
            pview = as_strided(p, shape=(
                p.shape[0] - borders[0],
                p.shape[1] - borders[1]) + window, strides=p.strides * 2)
            bview = as_strided(b, shape=(
                b.shape[0] - borders[0],
                b.shape[1] - borders[1]) + window, strides=b.strides * 2)
            psum = np.einsum('ijkl->ij', pview)
            bsum = np.einsum('ijkl->ij', bview)
            output[..., plane] = psum // bsum

        # At this point output should contain a reasonably "normal" looking
        # image, although it still won't look as good as the camera's normal
        # output (as it lacks vignette compensation, AWB, etc).
        #
        # If you want to view this in most packages (like GIMP) you'll need to
        # convert it to 8-bit RGB data. The simplest way to do this is by
        # right-shifting everything by 2-bits (yes, this makes all that
        # unpacking work at the start rather redundant...)

        return (output >> 2).astype(np.uint8)

    def get_image(self):
        ret_val, img = self.camera.read()
        if ret_val:
            return img
        
        # Something is wrong
        
        return None
    
    def __del__(self):
        self.camera.close()