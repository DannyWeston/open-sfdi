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

def AlwaysNumpy(arr):
    if type(arr) == np.ndarray:
        return arr
    
    with ProcessingContext.UseGPU(True):
        return arr.get()
    
    # We should only get to this point if the type provided is incorrect
    raise Exception("Incorrect type provided (must be numpy or cupy array)")