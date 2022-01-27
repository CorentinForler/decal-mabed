from contextlib import contextmanager
import io


@contextmanager
def profile(name: str = None):
    import cProfile
    import pstats
    import os

    pr = cProfile.Profile()
    pr.enable()

    yield

    pr.disable()

    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    if name is not None:
        with open(os.path.join('/tmp', 'python-profiler-' + name + '.txt'), 'w') as f:
            f.write(s.getvalue())

    print(s.getvalue())
