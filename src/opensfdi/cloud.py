import struct
import numpy as np
import open3d as o3d

from pathlib import Path

from .devices import BaseCamera

from . import characterisation, utils, image


def AlignToCalibBoard(pc: np.ndarray, cam: BaseCamera, board: characterisation.CalibrationBoard):
    centreCoords = board.GetBoardCentreCoords()

    # Compute inverse transform (camera to checkerboard)
    R = cam.characterisation.rotation.T

    t = -R @ cam.characterisation.translation

    # Shift origin to checkerboard center
    # t = t - R @ centreCoords

    return (R @ (pc.T + t)).T

def ArrayToCloud(np_cloud: np.ndarray, texture=None):
    xp = utils.ProcessingContext().xp

    # Filter out any NaNs from the point cloud and dc_img
    nan_points = xp.any(xp.isnan(np_cloud), axis=2)
    valid_points = xp.bitwise_not(nan_points)
    np_cloud = np_cloud[valid_points]
    texture = texture[valid_points]

    # open3d needs to use CPU context only
    with utils.ProcessingContext.UseGPU(False):
        xp = utils.ProcessingContext().xp

        np_cloud = utils.ToContext(xp, np_cloud)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np_cloud)

        if texture is not None: 
            pc = SetCloudColours(pc, texture)

        return pc

def FilterCloud(cloud, x=None, y=None, z=None):
    xp = utils.ProcessingContext().xp

    # Stop pointless alloc below :)
    if all((x is None, y is None, z is None)):
        return cloud

    mask = xp.ones(cloud.shape[:-1], dtype=xp.bool_)

    if x: mask = mask & image.ThresholdMask(cloud[..., 0], x[0], x[1])

    if y: mask = mask & image.ThresholdMask(cloud[..., 1], y[0], y[1])

    if z: mask = mask & image.ThresholdMask(cloud[..., 2], z[0], z[1])

    return mask

def LoadCloud(filename):
    return o3d.io.read_point_cloud(filename)

def SetCloudColours(pc, colours):
    # open3d requires textures to be colour-based (3 channel), and RGB (not BGR like opencv)
    # doesn't support cupy arrays too (GPU-side)
    with utils.ProcessingContext.UseGPU(False):
        xp = utils.ProcessingContext().xp

        colours = utils.ToContext(xp, colours)

        if colours.ndim == 1:
            coloursRGB = colours.reshape(-1, 1) * np.ones((1, 3))
        else:
            coloursRGB = np.flip(colours, axis=1)

        pc.colors = o3d.utility.Vector3dVector(coloursRGB)

    return pc

def DrawClouds(pointclouds):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for pc in pointclouds:
        vis.add_geometry(pc)

    # Add coordinate axis

    vis.run()
    vis.destroy_window()

def DrawCloud(cloud):
    o3d.visualization.draw_geometries([cloud])

def LoadMesh(filepath):
    return o3d.io.read_triangle_mesh(filepath)

def MeshToCloud(mesh, samples=10000):
   return mesh.sample_points_poisson_disk(samples)

def SaveCloud(filename: Path, point_cloud):
    o3d.io.write_point_cloud(filename, point_cloud)

def SaveArrayAsCloud(filename: Path, data):
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