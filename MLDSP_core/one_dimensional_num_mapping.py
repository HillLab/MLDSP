"""
@Daniel
"""
from pathlib import Path
from string import ascii_uppercase
from typing import Callable, Tuple, Any

from numpy import ndarray, array, vectorize, where, zeros, save, abs
from pyfaidx import FastaRecord
from pywt import pad
from scipy import fft

ZERO_MAP = ascii_uppercase.translate({65: '', 84: '', 71: '', 67: ''})
ZERO_MAP = dict(zip(ZERO_MAP, [0] * len(ZERO_MAP)))


def num_mapping_AT_CG(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    mapping = dict(**{'A': 1, 'C': -1, 'G': -1, 'T': 1}, **ZERO_MAP)
    return vectorize(mapping.get)(sq)


def num_mapping_justA(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    return where(sq == 'A', 1, 0)


def num_mapping_justC(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    return where(sq == 'C', 1, 0)


def num_mapping_justG(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    return where(sq == 'G', 1, 0)


def num_mapping_justT(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    return where(sq == 'T', 1, 0)


def num_mapping_Real(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    mapping = dict(**{'A': -1.5, 'C': 0.5, 'G': -0.5, 'T': 1.5}, **ZERO_MAP)
    return vectorize(mapping.get)(sq)


def num_mapping_PP(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    mapping = dict(**{'A': -1, 'C': 1, 'G': -1, 'T': 1}, **ZERO_MAP)
    return vectorize(mapping.get)(sq)


def num_mapping_IntN(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    dob = dict(**{'T': 1, 'C': 2, 'A': 3, 'G': 4}, **ZERO_MAP)
    return vectorize(dob.get)(sq)


def num_mapping_Int(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    dob = dict(**{'T': 0, 'C': 1, 'A': 2, 'G': 3}, **ZERO_MAP)
    return vectorize(dob.get)(sq)


def num_mapping_EIIP(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    mapping = dict(**{'A': 0.1260, 'C': 0.1340, 'G': 0.0806, 'T': 0.1335},
                   **ZERO_MAP)
    return vectorize(mapping.get)(sq)


def num_mapping_Atomic(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    mapping = dict(**{"A": 70, "C": 58, "G": 78, "T": 66},
                   **ZERO_MAP)
    return vectorize(mapping.get)(sq)


def num_mapping_Codons(sq: ndarray) -> ndarray:
    """
    @Daniel
    Args:
        sq:

    Returns:

    """
    # Authored by Wanxin Li @wxli0
    sq = ''.join(sq)
    length = len(sq)
    numSeq = zeros(length)
    codons = ['TTT', 'TTC', 'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG',
              'TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC', 'TAT', 'TAC',
              'TAA', 'TAG', 'TGA', 'TGT', 'TGC', 'TGG', 'CCT', 'CCC',
              'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC',
              'CGA', 'CGG', 'AGA', 'AGG', 'ATT', 'ATC', 'ATA', 'ATG',
              'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAA', 'AAG',
              'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG',
              'GAT', 'GAC', 'GAA', 'GAG', 'GGT', 'GGC', 'GGA', 'GGG']

    for idx in range(length):
        if idx <= (length - 3):
            t = sq[idx:idx + 3]
        elif idx == (length - 2):
            t = sq[idx:idx + 2] + sq[0:1]
        else:
            t = sq[idx] + sq[0:2]
        tp = codons.index(t)
        numSeq[idx] = tp
    return numSeq


def num_mapping_Doublet(sq: ndarray) -> ndarray:
    # Authored by Wanxin Li @wxli0
    """computes Doublet representation
    Keyword arguments:
    sq: sequence
    """
    sq = ''.join(sq)
    sq_len = len(sq)
    doublet = ['AA', 'AT', 'TA', 'AG', 'TT', 'TG', 'AC', 'TC', 'GA',
               'CA', 'GT', 'GG', 'CT', 'GC', 'CG', 'CC']
    numSeq = zeros(len(sq))
    # TODO: remove alpha for now, if alpha is added, then Codons also needs to be updated

    for idx in range(sq_len):
        if idx < (sq_len - 1):
            t = sq[idx:idx + 2]
        else:
            t = sq[idx] + sq[0]
        tp = doublet.index(t)
        numSeq[idx] = tp
    return numSeq


def one_dimensional_num_mapping_wrapper(
        seq: FastaRecord, method: Callable, results_path: Path,
        med_len: int = 100) -> Tuple[Any, Any, None]:
    """
    @Daniel
    Args:
        seq:
        method:
        results_path:
        med_len:

    Returns:

    """
    # normalize sequences to median seq length of cluster
    seq_new = str(seq).upper()
    name = seq.name
    if len(seq_new) >= med_len:
        seq_new = seq_new[0:round(med_len)]
    num_seq = method(array(list(seq_new)))
    if len(num_seq) < med_len:
        pad_width = int(med_len - len(num_seq))
        num_seq = pad(num_seq, pad_width, 'antisymmetric')[pad_width:]
    ofname = results_path.joinpath('Num_rep', f'{str(method).split()[1]}_{name}').resolve
    save(str(ofname), num_seq)
    fft_output = fft.fft(num_seq)
    abs_fft_output = abs(fft_output.flatten())
    return abs_fft_output, fft_output, None
