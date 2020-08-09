import numpy as np

def numMappingEIIP(sq):
    """computes EIIP representation

    Keyword arguments:
    sq: sequence
    """
    numSeq = np.zeros(len(sq))
    for idx, val in enumerate(sq):
        if val == 'A':
            numSeq[idx] = 0.1260
        elif val == 'C':
            numSeq[idx] = 0.1340
        elif val == 'G':
            numSeq[idx] = 0.0806
        elif val == 'T':
            numSeq[idx] = 0.1335
        return numSeq
