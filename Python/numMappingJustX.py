import numpy as np

def numMappingJustX(sq, X):
    """computes JustX representation

    Keyword arguments:
    sq: sequence
    X: one of 'A', 'C', 'G', 'T'
    """
    numSeq = np.zeros(len(sq))

    for idx, val in enumerate(sq):
        numSeq[idx] = (val == X) ? 1: 0
    return numSeq
