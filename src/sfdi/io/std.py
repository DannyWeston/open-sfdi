import os
import sys

from contextlib import contextmanager

from abc import ABC, abstractmethod

# Serialise/Deserialize libraries
class Serializable(ABC):
    @abstractmethod
    def serialize(self):
        raise NotImplementedError
    
    @abstractmethod
    def deserialize(self):
        raise NotImplementedError

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