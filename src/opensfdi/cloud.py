import struct
import numpy as np
import open3d as o3d

from pathlib import Path

from .devices import camera, characterisation

from . import image, utils


def AlignToCalibBoard(pc: np.ndarray, cam: camera.Camera, board: characterisation.CalibrationBoard):
    centreCoords = board.GetBoardCentreCoords()

    # Compute inverse transform (camera to checkerboard)
    R = cam.characterisation.rotation.T

    t = -R @ cam.characterisation.translation

    # Shift origin to checkerboard center
    # t = t - R @ centreCoords

    return (R @ (pc.T + t)).T

def ArrayToCloud(data: np.ndarray, colours=None):
    data = utils.ToNumpy(data)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data)

    if colours is not None: 
        pc = SetCloudColours(pc, colours)

    return pc

def LoadCloud(filename):
    return o3d.io.read_point_cloud(filename)

def SetCloudColours(pc, colours):
    # open3d requires textures to be colour-based (3 channel), and RGB (not BGR like opencv)
    # doesn't support cupy arrays too (GPU-side)
    colours = utils.ToNumpy(colours)

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

def SaveArrayAsCloud(filename: Path, data: np.ndarray):
    data = utils.ToNumpy(data)

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