import numpy as np

def numMappingCodons(sq, X):
    """computes Codon representation

    Keyword arguments:
    sq: sequence
    """
    sqLen = len(sq)
    numSeq = np.zeros(sqLen)
    codons = ['TTT','TTC','TTA','TTG','CTT','CTC','CTA','CTG','TCT','TCC','TCA','TCG','AGT','AGC','TAT','TAC',
              'TAA','TAG','TGA','TGT','TGC','TGG','CCT','CCC','CCA','CCG','CAT','CAC','CAA','CAG','CGT','CGC',
              'CGA','CGG','AGA','AGG','ATT','ATC','ATA','ATG','ACT','ACC','ACA','ACG','AAT','AAC','AAA','AAG',
              'GTT','GTC','GTA','GTG','GCT','GCC','GCA','GCG','GAT','GAC','GAA','GAG','GGT','GGC','GGA','GGG']

    for idx in range(sqLen):
        if idx <= (sqLen-3):
            t = sq[idx, idx+2]
        elif idx == (sqLen-2):
            t = sq[idx, idx+1]+sq[0:1]
        else:
            t = sq[idx]+sq[0:2]
        tp = codons.index(t)
        numSeq[idx] = tp
    return numSeq
