import numpy as np

def cgr(chars, order, k):
    """computes CGR representation in standard format: C top-left, G top-right, A bottom-left, T bottom-right

    Keyword arguments:
    chars: sequence
    order: chars to include in CGR
    k: value of k-mer
    """
    # set a numpy array of size 2^k,2^k  # remember that arrays are numbered top to bottom & left to right, unlike coordinate plots which go bottom up and left to right
    out = np.zeros((2**k,2**k))
    # set starting point of cgr plotting in the middle of cgr (x,y)
    x = 2**(k-1) 
    y = 2**(k-1)

    for i in range(len(chars)):
        char = chars[i]
        # devide x coordiate in half, moving it halfway to the left, this is correct if base is C or A
        x = int(x/2)
        # check to see if base is actually a G or T
        if char == order[2] or char == order[3]:  # if the nucleotide is G or T
            # add 2^(k-1) aka half the cgr length to the x value, brining it from 1/4 to 3/4
            x += 2**(k-1)
        # devide y coordiate in half, moving it halfway to the top, this is correct if base is C or G
        y = int(y/2)
        if char == order[0] or char == order[3]:  # if the nucleotide is A or T
            # add 2^(k-1) aka half the cgr length to the y value, brining it from 1/4 to 3/4
            y += 2**(k-1)
        # if i+1 is greater than or equal to k (i.e. if the position of the base is greater than k )
        if (i+1) >= k:
            # add plus 1 to the positions y & x in the cgr array
            out[y][x] += 1
    return out
