import numpy as np

def complete_mapping(nIn, nBits):
    mapping = np.arange(nIn)
    np.random.shuffle(mapping)
    mapping = np.array([mapping[i:i+nBits] for i in range(0, nIn, nBits)])
    return mapping

def mapping(nIn, nBits):
    return np.random.choice(np.arange(nIn), nBits, replace=False)