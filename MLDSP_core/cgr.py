"""
Functions to compute the CGR representation. This function assumes that
the sequence is in standard ACTG, with no missing values (N), modified
nucleotides or uracil
"""
import numpy as np


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
