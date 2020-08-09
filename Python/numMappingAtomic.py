import numpy as np

def numMappingAtomic(sq):
    """computes atomic representation

    Keyword arguments:
    sq:
    """
    sqLen = len(sq)
    numSeq = np.zeros((1, sqLen))
    for idx, val in enumerate(sq):
        if val == 'A':
            numSeq[idx] = 70
        elif val == 'C':
            numSeq[idx] = 58
        elif val == 'G':
            numSeq[idx] = 78
        elif val == 'T':
            numSeq[idx] = 66
