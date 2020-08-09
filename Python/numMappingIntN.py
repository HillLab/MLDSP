import numpy as np

def numMappingIntN(sq):
    """computes Integer (other variant) representation

    Keyword arguments:
    sq: sequence
    """
    dob = 'TCAG'
    numSeq = np.zeros(len(sq))

    for idx, val in enumerate(sq):
        numSeq[idx] = dob.find(val) #TODO: check -1

    return numSeq
