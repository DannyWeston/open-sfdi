import numpy as np

from matplotlib import pyplot as plt

def show_surface(data):
    hf = plt.figure()

    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))

    ha.plot_surface(X, Y, data)

    plt.show()

def show_heightmap(heightmap, title='Heightmap'):
    x, y = np.meshgrid(range(heightmap.shape[0]), range(heightmap.shape[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, np.transpose(heightmap))
    plt.title(title)
    plt.show()