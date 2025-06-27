import os
import sys
import numpy as np

from contextlib import contextmanager

# Redirect stdout to /dev/null
@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)

#     def to_stl(self, heightmap):
#         # Create vertices from the heightmap
#         vertices = []
#         for y in range(heightmap.shape[0]):
#             for x in range(heightmap.shape[1]):
#                 vertices.append([x, y, heightmap[y, x]])

#         vertices = np.array(vertices)

#         # Create faces for the mesh
#         faces = []
#         for y in range(heightmap.shape[0] - 1):
#             for x in range(heightmap.shape[1] - 1):
#                 v1 = x + y * heightmap.shape[1]
#                 v2 = (x + 1) + y * heightmap.shape[1]
#                 v3 = x + (y + 1) * heightmap.shape[1]
#                 v4 = (x + 1) + (y + 1) * heightmap.shape[1]

#                 # First triangle
#                 faces.append([v1, v2, v3])
#                 # Second triangle
#                 faces.append([v2, v4, v3])

#         # Create the mesh object
#         # mesh_data = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
#         # for i, f in enumerate(faces):
#         #     for j in range(3):
#         #         mesh_data.vectors[i][j] = vertices[f[j]]

#         # mesh_data.save('heightmap_mesh.stl')

def makeGreyFringes(frequency, phase, orientation, resolution=(1024, 1024)):
    w, h = resolution
    
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    g = np.sin(orientation) * x - np.cos(orientation) * y

    return (np.cos((2.0 * np.pi * g * frequency) - phase) + 1.0) / 2.0

def makeRGBFringes(frequency, phase, orientation, resolution=(1024, 1024), rgb=[1.0, 1.0, 1.0]):
    w, h = resolution

    img = np.empty((3, h, w), dtype=np.float32)

    img[2] = makeGreyFringes(frequency, phase, orientation, resolution) * rgb[0]
    img[1] = makeGreyFringes(frequency, phase, orientation, resolution) * rgb[1]
    img[0] = makeGreyFringes(frequency, phase, orientation, resolution) * rgb[2]

    return img