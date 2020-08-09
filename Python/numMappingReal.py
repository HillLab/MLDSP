import numpy as np

def numMappingReal(sq):
    """computes Real representation

    Keyword arguments:
    sq: sequence
    """
    numSeq = np.zeros(len(sq))
    for idx, val in enumerate(sq):
        if val == 'A':
            numSeq[idx] = -1.5
        elif val == 'C':
            numSeq[idx] = 0.5
        elif val == 'G':
            numSeq[idx] = -0.5
        elif val == 'T':
            numSeq[idx] = 1.5
        return numSeq
