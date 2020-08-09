import numpy as np 

def numMappingDoublet(sq):
    """computes Doublet representation

    Keyword arguments:
    sq: sequence
    """
    sqLen = len(sq)
    doublet = ['AA','AT','TA','AG','TT','TG','AC','TC','GA','CA','GT','GG','CT','GC','CG','CC']
    numSeq = np.zeros(len(sq))
    # alpha = 0 # TODO: remove alpha for now, if alpha is added, then Codons also needs to be updated
    kStrings = (2*alpha)+1
    
    for idx in range(sqLen):
        # if alpha == 0:
        if idx < (sqLen-1):
            t = sq[idx, idx+2]
        else:
            t = sq[idx]+sq[0]
        tp = doublet.index(t)
        numSeq[idx] = tp
    return numSeq
