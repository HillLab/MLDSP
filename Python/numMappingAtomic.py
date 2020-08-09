import numpy as np

def numMappingAtomic(sq):
    """computes Atomic representation

    Keyword arguments:
    sq: sequence
    """
    numSeq = np.zeros(len(sq))
    for idx, val in enumerate(sq):
        if val == 'A':
            numSeq[idx] = 70
        elif val == 'C':
            numSeq[idx] = 58
        elif val == 'G':
            numSeq[idx] = 78
        elif val == 'T':
            numSeq[idx] = 66
        return numSeq
