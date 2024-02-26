import numpy as np

from matplotlib import pyplot as plt
from numpy.polynomial import polynomial as P

from abc import ABC

from sfdi import wrapped_phase, unwrapped_phase

def show_heightmap(heightmap, title='Heightmap'):
    x, y = np.meshgrid(range(heightmap.shape[0]), range(heightmap.shape[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, np.transpose(heightmap))
    plt.title(title)
    plt.show()

class PhaseHeight(ABC):    
    def phasemap(self, imgs):
        w_phase = wrapped_phase(imgs)
        return unwrapped_phase(w_phase)

class ClassicPhaseHeight(PhaseHeight):
    # ℎ = 𝜙𝐷𝐸 ⋅ 𝑝 ⋅ 𝑑 / 𝜙𝐷𝐸 ⋅ 𝑝 + 2𝜋𝑙
    # p = stripe width
    # d = distance between camera and reference plane
    # l = distance between camera and projector
        
    def __init__(self, p, d, l):
        super().__init__()
        
        self.p = p
        self.d = d 
        self.l = l
    
    def heightmap(self, ref_imgs, imgs):
        ref_phase, measured_phase = self.phasemap(ref_imgs), self.phasemap(imgs)

        phase_diff = measured_phase - ref_phase
        
        heightmap = np.divide(phase_diff * self.p * self.d, phase_diff * self.p + (2.0 * np.pi * self.l))
        
        #heightmap[heightmap <= 0] = 0 # Remove negative values

        heightmap = heightmap - heightmap[0][0] # remove ref plane value

        return heightmap

class TriangularStereoHeight(PhaseHeight):
    def __init__(self, ref_dist, sensor_dist, freq):
        super().__init__()
        
        self.ref_dist = ref_dist
        self.sensor_dist = sensor_dist
        self.freq = freq
    
    def heightmap(self, imgs):
        phase = self.phasemap(imgs)

        #heightmap = np.divide(self.ref_dist * phase_diff, 2.0 * np.pi * self.sensor_dist * self.freq)
        
        #heightmap[heightmap <= 0] = 0 # Remove negative values

        return None

class PolyPhaseHeight(PhaseHeight):
    def __init__(self, coeffs=None):
        super().__init__()
        
        self.coeffs = coeffs
    
    def calibrate(self, heightmap, ref_imgs, imgs, deg=1):
        ref_phase, measured_phase = self.phasemap(ref_imgs), self.phasemap(imgs)
        
        diff = ref_phase - measured_phase

        total = np.zeros(ref_phase.shape, dtype=np.float64)
        
        for i in range(deg):
            total += np.power(diff, i)
            
        # Coefficients are in ascending order

        self.coeffs, stats = P.polyfit(diff.ravel(), heightmap.ravel(), deg=deg, full=True)
            
        return self.coeffs, stats[0][0]

    def heightmap(self, ref_imgs, imgs):
        ref_phase, measured_phase = self.phasemaps(ref_imgs, imgs)
        
        diff = ref_phase - measured_phase
        
        result = np.zeros(ref_phase.shape, dtype=np.float64)
        
        for i, a_i in enumerate(self.coeffs):
            print(f'{round(a_i, ndigits=3)} X_{i}')
            result += (np.power(diff, i) * a_i)
        
        return result