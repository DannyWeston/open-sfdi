import numpy as np
from matplotlib import pyplot as plt

def show_heightmap(heightmap, title='Heightmap'):
    x, y = np.meshgrid(range(heightmap.shape[0]), range(heightmap.shape[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, np.transpose(heightmap))
    plt.title(title)
    plt.show()

class PhaseHeight:
    def __init__(self):
        pass 

    def heightmap(self, phasemaps: list):
        pass
    
class ClassicPhaseHeight(PhaseHeight):
    def __init__(self, p, d, l):
        super().__init__()
        
        self.p = p
        self.d = d
        self.l = l
    
    def heightmap(self, ref_phase, measured_phase):
        phase_diff = measured_phase - ref_phase
        
        heightmap = np.divide(phase_diff * self.p * self.d, phase_diff * self.p + (2.0 * np.pi * self.l))
        
        #heightmap[heightmap <= 0] = 0

        return heightmap