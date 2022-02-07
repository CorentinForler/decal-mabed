from contextlib import contextmanager
import io


@contextmanager
def profile(name: str = None):
    """Profile a block of code.

    Args:
        name (str, optional): If present, create a file in /tmp with the contents of the profile. Defaults to None.
    """
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
        with open(os.path.join('/tmp', 'python-profiler-' + name + '.txt'), 'w', encoding='utf-8') as f:
            f.write(s.getvalue())

    print(s.getvalue())
