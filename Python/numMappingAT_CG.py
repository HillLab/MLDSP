def numMappingAT_CG(sq):
    """computes paired numeric representation

    Keyword arguments:
    sq: sequence
    """
    numSeq = np.zeros(len(sq))
    for idx, val in enumerate(sq):
        if val == 'A':
            numSeq[idx] = -1
        elif val == 'C':
            numSeq[idx] = -1
        elif val == 'G':
            numSeq[idx] = -1
        elif val == 'T':
            numSeq[idx] = -1
        return numSeq
