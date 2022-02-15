"""
Functions to compute the CGR representation. This function assumes that
the sequence is in standard ACTG, with no missing values (N), modified
nucleotides or uracil
"""
from pathlib import Path
from typing import Tuple, Any

import numpy as np
from pyfaidx import FastaRecord
from scipy import fft


def cgr(chars: str, order: str = "ACGT", k: int = 6):
    """
    computes CGR representation in standard format: C top-left,
    G top-right, A bottom-left, T bottom-right

    params chars: sequence
    params order: chars to include in CGR
    params k: value of k-mer
    """
    size = 2 ** k
    mid_cgr = 2 ** (k - 1)
    chaos = np.zeros((size, size))
    # set starting point of cgr plotting in the middle of cgr (x,y)
    x = y = mid_cgr

    for i in range(len(chars)):
        char = chars[i]
        # divide x coordinate in half, moving it halfway to the left, this is correct if base is C or A
        x //= 2
        # check to see if base is actually a G or T
        if char == order[2] or char == order[3]:
            # add 2^(k-1) aka half the cgr length to the x value, brining it from 1/4 to 3/4
            x += mid_cgr
        # divide y coordinate in half, moving it halfway to the top, this is correct if base is C or G
        y //= 2
        if char == order[0] or char == order[3]:
            # add 2^(k-1) aka half the cgr length to the y value, brining it from 1/4 to 3/4
            y += mid_cgr
        # if i+1 is greater than or equal to k (i.e. if the position of the base is greater than k )
        if (i + 1) >= k:
            # add plus 1 to the positions y & x in the cgr array
            chaos[y][x] += 1
    return chaos


def PuPyCGR(seq: FastaRecord, kmer: int, results: Path,
            order: str = 'ACGT') -> Tuple[Any, Any, Any]:  # TODO change any for the actual signature)
    """
    Wrapper of CGR to compute PuPyCGR
    Args:
        seq: FastaRecord instance with the sequence and name of the sequence
        kmer: Kmer value to use
        results: Path to the results folder
        order: Order of the nucleotides in the Chaos square


    Returns:
    """
    return compute_cgr(seq=seq, kmer=kmer, results=results, order=order,
                       pyrimidine=True)


def oneDPuPyCGR(seq: FastaRecord, kmer: int, results: Path,
                order: str = 'ACGT') -> Tuple[Any, Any, Any]:  # TODO change any for the actual signature)
    """
    Wrapper of CGR to compute 1DPuPyCGR
    Args:
        seq: FastaRecord instance with the sequence and name of the sequence
        kmer: Kmer value to use
        results: Path to the results folder
        order: Order of the nucleotides in the Chaos square


    Returns:
    """
    return compute_cgr(seq=seq, kmer=kmer, results=results, order=order,
                       pyrimidine=True, last_only=True)


def compute_cgr(seq: FastaRecord, results: Path, kmer: int = 5,
                order: str = 'ACGT', pyrimidine: bool = False,
                last_only: bool = False, **kwargs) -> Tuple[Any, Any, Any]:  # TODO change any for the actual signature
    """
    This function compute the CGR matrix for a sequence in seq_dict
    Args:
        last_only: takes only the last (bottom) row but all columns of cgr to make 1DPuPyCGR
        pyrimidine: Replace purine for pyrimidines (PuPyCGR, 1DPuPyCGR)
        seq: FastaRecord instance with the sequence and name of the sequence
        kmer: Kmer value to use
        results: Path to the results folder
        order: Order of the nucleotides in the Chaos square

    Returns:

    """
    seq_new = str(seq)
    name = seq.name
    # Replace complementary Purine/Pyrimidine
    if pyrimidine:
        seq_new = seq_new.replace('G', 'A').replace('C', 'T')
    cgr_raw = cgr(seq_new, order, kmer)
    if last_only:
        cgr_out = cgr_raw[-1, :]
    else:
        cgr_out = cgr_raw
    # shape:[2^k, 2^k] # may not be appropriate to take by column
    out_filename = str(results.joinpath(
        'Num_rep', f'cgr_k={kmer}_{name}').resolve())
    np.save(out_filename, cgr_out)
    fft_out = fft.fft(cgr_out, axis=0)
    abs_fft_out = np.abs(fft_out.flatten())
    return abs_fft_out, fft_out, cgr_out  # flatted into 1d array
