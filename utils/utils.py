import numpy as np

def joint_shuffle(*args, seed=None):
    
    if seed != None:
        np.random.seed(seed)
    
    indexs = np.arange(args[0].shape[0])
    np.random.shuffle(indexs)
    
    shuffled_args = []

    for arg in args:
        shuffled_args.append(arg[indexs])
    
    return shuffled_args