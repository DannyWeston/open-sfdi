import struct
import numpy as np
import open3d as o3d

from .devices import camera, board
from pathlib import Path

def AlignToCalibBoard(pc: np.ndarray, cam: camera.Camera, board: board.CalibrationBoard):
    centreCoords = board.GetBoardCentreCoords()

    # Compute inverse transform (camera to checkerboard)
    R = cam.visionConfig.rotation.T

    t = -R @ cam.visionConfig.translation

    # Shift origin to checkerboard center
    # t = t - R @ centreCoords

    return (R @ (pc.T + t)).T

def NumpyToCloud(data: np.ndarray):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data)

    return pc

def LoadCloud(filename):
    return o3d.io.read_point_cloud(filename)

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

def SaveNumpyAsCloud(filename: Path, arr: np.ndarray):
  with open(filename, "wb") as file:
    file.write(bytes('ply\n', 'utf-8'))
    file.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    file.write(bytes(f'element vertex {arr.shape[0]}\n', 'utf-8'))
    file.write(bytes(f'property float x\n', 'utf-8'))
    file.write(bytes(f'property float y\n', 'utf-8'))
    file.write(bytes(f'property float z\n', 'utf-8'))
    file.write(bytes(f'end_header\n', 'utf-8'))

    for i in range(arr.shape[0]):
      file.write(bytearray(struct.pack("fff", arr[i, 0], arr[i, 1], arr[i, 2])))