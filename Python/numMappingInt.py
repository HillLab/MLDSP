import numpy as np

def numMappingInt(sq):
    """computes Integer representation

    Keyword arguments:
    sq: sequence
    """
    dob = 'TCAG'
    numSeq = np.zeros(len(sq))

    for idx, val in enumerate(sq):
        numSeq[idx] = dob.find(val) #TODO: check -1

    return numSeq
