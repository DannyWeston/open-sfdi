import os
import sys

from contextlib import contextmanager

# Redirect stdout from Blender to null as we don't want it
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