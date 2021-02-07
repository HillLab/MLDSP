import numpy as np


def num_mapping_AT_CG(sq):
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        if t == 'A':
            numSeq[k] = 1
        elif t == 'C':
            numSeq[k] = -1
        elif t == 'G':
            numSeq[k] = -1
        elif t == 'T':
            numSeq[k] = 1
        else:
            pass
    return numSeq


def num_mapping_justA(sq):
    a = "A"
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        if t.upper() == a:
            numSeq[k] = 1
        else:
            pass
    return numSeq


def num_mapping_justC(sq):
    c = "C"
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        if t.upper() == c:
            numSeq[k] = 1
        else:
            pass
    return numSeq


def num_mapping_justG(sq):
    g = "G"
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        if t.upper() == g:
            numSeq[k] = 1
        else:
            pass
    return numSeq


def num_mapping_justT(sq):
    t_ = "T"
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        if t.upper() == t_:
            numSeq[k] = 1
        else:
            pass
    return numSeq


def num_mapping_Real(sq):
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        if t.upper() == "A":
            numSeq[k] = -1.5
        elif t.upper() == "C":
            numSeq[k] = 0.5
        elif t.upper() == "G":
            numSeq[k] = -0.5
        elif t.upper() == "T":
            numSeq[k] = 1.5
        else:
            pass
    return numSeq


def num_mapping_PP(sq):
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        if t.upper() == "A":
            numSeq[k] = -1
        elif t.upper() == "C":
            numSeq[k] = 1
        elif t.upper() == "G":
            numSeq[k] = -1
        elif t.upper() == "T":
            numSeq[k] = 1
        else:
            pass
    return numSeq


def num_mapping_IntN(sq):
    dob = ['T', 'C', 'A', 'G']
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        tp = dob.index(t) + 1
        numSeq[k] = tp
    return numSeq


def num_mapping_Int(sq):
    dob = ['T', 'C', 'A', 'G']
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        tp = dob.index(t)
        numSeq[k] = tp
    return numSeq


def num_mapping_EIIP(sq):
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        if t.upper() == "A":
            numSeq[k] = 0.1260
        elif t.upper() == "C":
            numSeq[k] = 0.1340
        elif t.upper() == "G":
            numSeq[k] = 0.0806
        elif t.upper() == "T":
            numSeq[k] = 0.1335
        else:
            pass
    return numSeq


def num_mapping_Atomic(sq):
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        if t.upper() == "A":
            numSeq[k] = 70
        elif t.upper() == "C":
            numSeq[k] = 58
        elif t.upper() == "G":
            numSeq[k] = 78
        elif t.upper() == "T":
            numSeq[k] = 66
        else:
            pass
    return numSeq


def num_mapping_Codons(sq):
    # Authored by Wanxin Li @wxli0
    length = len(sq)
    numSeq = np.zeros(length)
    codons = ['TTT','TTC','TTA','TTG','CTT','CTC','CTA','CTG','TCT','TCC','TCA','TCG','AGT','AGC','TAT','TAC',
              'TAA','TAG','TGA','TGT','TGC','TGG','CCT','CCC','CCA','CCG','CAT','CAC','CAA','CAG','CGT','CGC',
              'CGA','CGG','AGA','AGG','ATT','ATC','ATA','ATG','ACT','ACC','ACA','ACG','AAT','AAC','AAA','AAG',
              'GTT','GTC','GTA','GTG','GCT','GCC','GCA','GCG','GAT','GAC','GAA','GAG','GGT','GGC','GGA','GGG']
    # for k in range(0, length):
    #     if k < length-1:
    #         t = sq[k:k+3]
    #     elif k == length-1:
    #         t = sq[k:k+2] + sq[0]
    #     else:
    #         t = sq[k] + sq[0:2]
    #     tp = codons.index(t)
    #     numSeq[k] = tp
    for idx in range(length):
        if idx <= (length-3):
            t = sq[idx:idx+3]
        elif idx == (length-2):
            t = sq[idx:idx+2]+sq[0:1]
        else:
            t = sq[idx]+sq[0:2]
        tp = codons.index(t)
        numSeq[idx] = tp
    return numSeq


def num_mapping_Doublet(sq):
    # Authored by Wanxin Li @wxli0
    """computes Doublet representation
    Keyword arguments:
    sq: sequence
    """
    sq_len = len(sq)
    doublet = ['AA', 'AT', 'TA', 'AG', 'TT', 'TG', 'AC',
               'TC', 'GA', 'CA', 'GT', 'GG', 'CT', 'GC', 'CG', 'CC']
    numSeq = np.zeros(len(sq))
    # alpha = 0 # TODO: remove alpha for now, if alpha is added, then Codons also needs to be updated

    for idx in range(sq_len):
        # if alpha == 0:
        if idx < (sq_len-1):
            t = sq[idx:idx+2]
        else:
            t = sq[idx]+sq[0]
        tp = doublet.index(t)
        numSeq[idx] = tp
    return numSeq
