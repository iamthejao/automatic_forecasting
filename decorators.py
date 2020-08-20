import functools
import time

def timeit(f):
    @functools.wraps(f)
    def timer(*args, **kwargs):
        start = time.perf_counter()    # 1
        return_ = f(*args, **kwargs)
        end = time.perf_counter()      # 2
        elapsed = end - start    # 3
        elapsed = elapsed * 1E3
        print(f"{f.__name__!r} took {elapsed:.6f} msecs")
        return return_
    return timer


def lazy(func):
    def lazyfunc(*args, **kwargs):
        wrapped = lambda : func(*args, **kwargs)
        wrapped.__name__ = "lazy-" + func.__name__
        return wrapped
    return lazyfunc