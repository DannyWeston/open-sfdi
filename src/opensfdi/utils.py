import os
import sys
import cProfile
import pstats
import numpy as np

from contextlib import contextmanager
from typing import ClassVar, Set

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

class SerialisableMixin:
    _type_registry: ClassVar[dict] = {}
    _exclude_fields: ClassVar[Set[str]] = set()  # Excludes
    
    def __init_subclass__(cls):
        cls._type_registry[cls.__name__] = cls
        
        # Merge parent's exclude fields with child's
        parent_excludes = getattr(super(cls, cls), '_exclude_fields', set())
        child_excludes = getattr(cls, '_exclude_fields', set())
        cls._exclude_fields = parent_excludes | child_excludes
        
        super().__init_subclass__()
    
    def to_dict(self) -> dict:
        """Convert to dict, excluding specified fields"""
        data = {}
        
        for key, value in self.__dict__.items():
            # Skip fields in exclude list
            if key in self._exclude_fields:
                continue

            # Remove trailing underscores
            if key.startswith('_'):
                key = key[1:]

            if hasattr(value, 'to_dict'):
                d = value.to_dict()

                # Recurse through serialisable children
                if isinstance(value, SerialisableMixin):
                    d["__type__"] = value.__class__.__name__
                
                data[key] = d

            elif isinstance(value, np.ndarray):
                data[key] = value.tolist()

            else:
                data[key] = value
        
        data['__type__'] = self.__class__.__name__
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Create object from dict"""
        type_name = data.pop('__type__')

        subclass = cls._type_registry[type_name]

        vars = dict()

        for key, value in data.items():
            if isinstance(value, dict) and "__type__" in value:
                # Found another SerialisableMixin, need to initialise correctly
                value = SerialisableMixin.from_dict(value)

            vars[key] = value

        return subclass(**vars)


def TransMat(R, T):
    M = np.eye(4, 4)
    M[:3, :3] = R
    M[:3, 3] = T
    return M

def ToContext(xp, arr):
    if isinstance(arr, xp.ndarray):
        return arr
    
    # Not matching type so convert
    if isinstance(arr, np.ndarray):
        xp.asarray(arr)
        
    return arr.get()

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