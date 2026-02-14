import struct
import numpy as np

from pathlib import Path

from vedo import Points, show

from .devices import BaseCamera
from . import characterisation, utils, image

# def AlignToCalibBoard(pc: np.ndarray, cam: BaseCamera, board: characterisation.CalibrationBoard):
#     centreCoords = board.GetBoardCentreCoords()

#     # Compute inverse transform (camera to checkerboard)
#     R = cam.characterisation.rotation.T

#     t = -R @ cam.characterisation.translation

#     # Shift origin to checkerboard center
#     # t = t - R @ centreCoords

#     return (R @ (pc.T + t)).T

def np_to_cloud(np_cloud: np.ndarray, texture=None):
    xp = utils.ProcessingContext().xp

    # Filter out any NaNs from the point cloud and dc_img    
    nan_points = xp.any(xp.isnan(np_cloud), axis=2)
    valid_points = xp.bitwise_not(nan_points)
    np_cloud = np_cloud[valid_points]

    if texture is not None:
        texture = texture[valid_points]
        texture = image.ToInt(np.column_stack([texture, texture, texture]))

    # Need array on CPU
    with utils.ProcessingContext.UseGPU(False):
        xp = utils.ProcessingContext().xp

        np_cloud = utils.ToContext(xp, np_cloud)
        texture = utils.ToContext(xp, texture)

        point_cloud = Points(np_cloud)

        if texture is not None:
            point_cloud.pointcolors = texture

        return point_cloud

def filter_np_cloud(cloud, x=None, y=None, z=None):
    xp = utils.ProcessingContext().xp

    # Stop pointless alloc below :)
    if all((x is None, y is None, z is None)):
        return cloud

    mask = xp.ones(cloud.shape[:-1], dtype=xp.bool_)

    if x: mask = mask & image.ThresholdMask(cloud[..., 0], x[0], x[1])

    if y: mask = mask & image.ThresholdMask(cloud[..., 1], y[0], y[1])

    if z: mask = mask & image.ThresholdMask(cloud[..., 2], z[0], z[1])

    return mask

def show_cloud(cloud: Points, point_size=2):
    show(cloud.ps(point_size))

# def LoadCloud(filename):
#     return o3d.io.read_point_cloud(filename)

def save_np_as_ply(filename: Path, data):
    with utils.ProcessingContext.UseGPU(False):
        xp = utils.ProcessingContext().xp

        data = utils.ToContext(xp, data)

        with open(filename, "wb") as file:
            file.write(bytes('ply\n', 'utf-8'))
            file.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
            file.write(bytes(f'element vertex {data.shape[0]}\n', 'utf-8'))
            file.write(bytes(f'property float x\n', 'utf-8'))
            file.write(bytes(f'property float y\n', 'utf-8'))
            file.write(bytes(f'property float z\n', 'utf-8'))
            file.write(bytes(f'end_header\n', 'utf-8'))

            for i in range(data.shape[0]):
                file.write(bytearray(struct.pack("fff", data[i, 0], data[i, 1], data[i, 2])))