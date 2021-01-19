import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

def dimReduction(data, n_dim, method):
    """
    Function will take in a nxm 2d-array and reduce the dimensions of the data using a specified dimensionality
    reduction technique (PCA, MDS, or TSNE).
    :param np.array data: input data to be transformed
    :param int n_dim: dimensions to reduce to
    :param str method: which method to use (either 'pca', 'mds', or 'tsne')
    :return np.array transformed: nxn_dim array of tranformed data
    """
    if method == 'pca':
        pca = PCA(n_components=n_dim)
        transformed = pca.fit_transform(data)
        return transformed
    elif method == 'mds':
        mds = MDS(n_components=n_dim)
        transformed = mds.fit_transform(data)
        return transformed
    elif method == 'tsne':
        tsne = TSNE(n_components=n_dim)
        transformed = tsne.fit_transform(data)
        return transformed
        