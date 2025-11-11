import os
import sys
import cProfile
import pstats

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

class ProcessingContext:
    __Instance = None
    
    def __new__(cls):
        if cls.__Instance is None:
            cls.__Instance = super().__new__(cls)
            cls.__Instance.UseGPU = False
            cls.__Instance.m_Processor = np

        return cls.__Instance
    
    @classmethod
    @contextmanager
    def UseGPU(cls, value=False):
        instance = cls()
        previous = instance.UseGPU
        
        try:
            instance.UseGPU = value
            yield instance

        finally:
            instance.UseGPU = previous

    @property
    def xp(self):
        if self.UseGPU:
            if self.m_Processor != np: return self.m_Processor
            
            try:
                import cupy as cp
                self.m_Processor = cp
                
                return self.m_Processor
            
            except ImportError:
                raise ImportError("GPU processing is not available as cupy is not installed")
        
        self.m_Processor = np
        return self.m_Processor

    def __str__(self):
        return f"ProcessingContext(UseGPU={self.UseGPU}"

def TransMat(R, T):
    M = np.eye(4, 4)
    M[:3, :3] = R
    M[:3, 3] = T
    return M

def ToNumpy(arr):
    # If type already matches, return
    if type(arr) == np.ndarray:
        return arr

    with ProcessingContext.UseGPU(True):
        xp = ProcessingContext().xp

        return xp.asnumpy(arr)
    
    # We should only get to this point if the type provided is incorrect
    raise Exception("Incorrect type provided (must be numpy or cupy array)")
    
def SingleBilinearInterp(img, coords):
    xp = ProcessingContext().xp

    fracs, integrals = xp.modf(coords)
    integrals = integrals.astype(xp.uint16)

    # a1 <-> a2 
    # ^      ^
    # |      |
    # v      v
    # a3 <-> a4
    a1 = img[integrals[0], integrals[1]]
    a2 = img[integrals[0], integrals[1] + 1]
    a3 = img[integrals[0] + 1, integrals[1]]
    a4 = img[integrals[0] + 1, integrals[1] + 1]
    
    x = (1 - fracs[0]) * ((1 - fracs[1]) * a1 + fracs[1] * a2) + fracs[0] * ((1 - fracs[1]) * a3 + fracs[1] * a4)

    return x

@contextmanager
def ProfileCode(depth=10):
    try:
        profiler = cProfile.Profile()
        profiler.enable()
        yield

    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(depth)