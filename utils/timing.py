import time
import numpy as np

def timeit(func, *args, **kwargs):
    a = time.time()
    out = func(*args, **kwargs)
    b = time.time()
    return b-a, out

def timeit_n(func, args=(), kwargs={}, n=1):
    outs = list()
    times = np.empty(n)
    
    for i in range(n):
        a = time.time()
        out = func(*args, **kwargs)
        b = time.time()
        
        times[i] = b-a
        outs.append(out)
    
    return times, outs