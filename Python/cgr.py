import numpy as np

def cgr(chars, order, k):
    """computes CGR representation

    Keyword arguments:
    chars: sequence
    order: chars to include in CGR
    k: value of k-mer
    """
    out = np.zeros((2**k,2**k))
    x = 2**(k-1)
    y = 2**(k-1)

    for i in range(len(chars)):
        char = chars[i]
        x = int(np.fix(x/2))
        if char == order[2] or char == order[3]:
            x += 2**(k-1)
        y = int(np.fix(y/2))
        if char == order[0] or char == order[3]:
            y += 2**(k-1)

        if (i+1) >= k:
            
            out[y][x] += 1
    return out
