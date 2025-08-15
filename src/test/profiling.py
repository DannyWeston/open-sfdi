import cProfile
import pstats

from contextlib import contextmanager
    
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