import numpy as np
from multiprocessing import Pool
from spicy import fft

def num_mapping_AT_CG(sq):
    """computes paired numeric representation
    Keyword arguments:
    sq: sequence
    """
    num_seq = np.zeros(len(sq))
    for idx, val in enumerate(sq):
        if val == 'A':
            num_seq[idx] = -1
        elif val == 'C':
            num_seq[idx] = -1
        elif val == 'G':
            num_seq[idx] = -1
        elif val == 'T':
            num_seq[idx] = -1
        return num_seq

import numpy as np

def num_mapping_Atomic(sq):
    """computes Atomic representation
    Keyword arguments:
    sq: sequence
    """
    num_seq = np.zeros(len(sq))
    for idx, val in enumerate(sq):
        if val == 'A':
            num_seq[idx] = 70
        elif val == 'C':
            num_seq[idx] = 58
        elif val == 'G':
            num_seq[idx] = 78
        elif val == 'T':
            num_seq[idx] = 66
        return num_seq

def num_mapping_Codons(sq, X):
    """computes Codon representation
    Keyword arguments:
    sq: sequence
    """
    sq_len = len(sq)
    num_seq = np.zeros(sq_len)
    codons = ['TTT','TTC','TTA','TTG','CTT','CTC','CTA','CTG','TCT','TCC','TCA','TCG','AGT','AGC','TAT','TAC',
              'TAA','TAG','TGA','TGT','TGC','TGG','CCT','CCC','CCA','CCG','CAT','CAC','CAA','CAG','CGT','CGC',
              'CGA','CGG','AGA','AGG','ATT','ATC','ATA','ATG','ACT','ACC','ACA','ACG','AAT','AAC','AAA','AAG',
              'GTT','GTC','GTA','GTG','GCT','GCC','GCA','GCG','GAT','GAC','GAA','GAG','GGT','GGC','GGA','GGG']

    for idx in range(sq_len):
        if idx <= (sq_len-3):
            t = sq[idx, idx+2]
        elif idx == (sq_len-2):
            t = sq[idx, idx+1]+sq[0:1]
        else:
            t = sq[idx]+sq[0:2]
        tp = codons.index(t)
        num_seq[idx] = tp
    return num_seq

def num_mapping_Doublet(sq):
    """computes Doublet representation
    Keyword arguments:
    sq: sequence
    """
    sq_len = len(sq)
    doublet = ['AA','AT','TA','AG','TT','TG','AC','TC','GA','CA','GT','GG','CT','GC','CG','CC']
    num_seq = np.zeros(len(sq))
    # alpha = 0 # TODO: remove alpha for now, if alpha is added, then Codons also needs to be updated
    
    for idx in range(sq_len):
        # if alpha == 0:
        if idx < (sq_len-1):
            t = sq[idx, idx+2]
        else:
            t = sq[idx]+sq[0]
        tp = doublet.index(t)
        num_seq[idx] = tp
    return num_seq

def num_mapping_EIIP(sq):
    """computes EIIP representation
    Keyword arguments:
    sq: sequence
    """
    num_seq = np.zeros(len(sq))
    for idx, val in enumerate(sq):
        if val == 'A':
            num_seq[idx] = 0.1260
        elif val == 'C':
            num_seq[idx] = 0.1340
        elif val == 'G':
            num_seq[idx] = 0.0806
        elif val == 'T':
            num_seq[idx] = 0.1335
        return num_seq

def num_mapping_Int(sq):
    """computes Integer representation
    Keyword arguments:
    sq: sequence
    """
    dob = 'TCAG'
    num_seq = np.zeros(len(sq))

    for idx, val in enumerate(sq):
        num_seq[idx] = dob.find(val) #TODO: check -1

    return num_seq

def num_mapping_IntN(sq):
    """computes Integer (other variant) representation
    Keyword arguments:
    sq: sequence
    """
    dob = 'TCAG'
    num_seq = np.zeros(len(sq))

    for idx, val in enumerate(sq):
        num_seq[idx] = dob.find(val) #TODO: check -1

    return num_seq

def num_mapping_justA(sq):
    """computes JustA representation
    Keyword arguments:
    sq: sequence
    """
    return num_mapping_justX(sq, 'A')

def num_mapping_justC(sq):
    """computes JustC representation
    Keyword arguments:
    sq: sequence
    """
    return num_mapping_justX(sq, 'C')

def num_mapping_justG(sq):
    """computes JustG representation
    Keyword arguments:
    sq: sequence
    """
    return num_mapping_justX(sq, 'G')

def num_mapping_justT(sq):
    """computes JustT representation
    Keyword arguments:
    sq: sequence
    """
    return num_mapping_justX(sq, 'T')

def num_mapping_justX(sq, X):
    """computes JustX representation
    Keyword arguments:
    sq: sequence
    X: one of 'A', 'C', 'G', 'T'
    """
    num_seq = np.zeros(len(sq))

    for idx, val in enumerate(sq):
        num_seq[idx] =  1 if (val == X) else 0
    return num_seq

def num_mapping_PP(sq):
    """computes Purine/Pyramidine representation
    Keyword arguments:
    sq: sequence
    """
    num_seq = np.zeros(len(sq))
    for idx, val in enumerate(sq):
        if val == 'A':
            num_seq[idx] = -1
        elif val == 'C':
            num_seq[idx] = 1
        elif val == 'G':
            num_seq[idx] = -1
        elif val == 'T':
            num_seq[idx] = 1
        return num_seq

def num_mapping_Real(sq):
    """computes Real representation
    Keyword arguments:
    sq: sequence
    """
    num_seq = np.zeros(len(sq))
    for idx, val in enumerate(sq):
        if val == 'A':
            num_seq[idx] = -1.5
        elif val == 'C':
            num_seq[idx] = 0.5
        elif val == 'G':
            num_seq[idx] = -0.5
        elif val == 'T':
            num_seq[idx] = 1.5
        return num_seq




def one_dnum_rep_mapping(seq, method_num, med_len, total_seq):
    pool = Pool()
    n_seq = []
    ns_list = np.zeros(total_seq)
    fft_output_list = np.zeros(total_seq)
    abs_fft_output_list = np.zeros(total_seq)
    def shorten_seq(seq_index):
        ns = seq[seq_index].upper()
        ind = med_len-len(ns)
        if ind < 0:
            seq[seq_index] = ns[:med_len+1]
    pool.map(shorten_seq, range(range(total_seq)))

    method_num_to_function = {2: num_mapping_PP, 3: num_mapping_Int, 4:num_mapping_IntN, 5: num_mapping_Real, 6:num_mapping_Real, 7:num_mapping_Codons, 8:num_mapping_Atomic, 9:num_mapping_EIIP, 10: num_mapping_AT_CG, 11: num_mapping_justA, 12: num_mapping_justC, 13: num_mapping_justG, 14: num_mapping_justT}
    def call_methods(seq_index):
        n_seq[seq_index] = method_num_to_function[method_num](seq[seq_index])
    pool.map(call_methods, range(total_seq))

    def extend_seq(seq_index):
        ns = n_seq[seq_index]
        ind = med_len - len(ns)
        
        if ind > 0:
            ns_list[seq_index] = np.pad(ns, ind, 'symmetric')[ind:] #TODO: anitisymmetric in Matlab
        else:
            ns_list[seq_index] = ns
        fft_output_list[seq_index] = fft(ns_list[seq_index])
        abs_fft_output_list[seq_index] = abs(fft_output_list[seq_index])
    pool.map(extend_seq, range(total_seq))
