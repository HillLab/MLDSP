from skbio.stats.ordination import pcoa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
        pca = PCA(n_components=n_dim,svd_solver='full')
        transformed = pca.fit_transform(data)
        return transformed
    #Not working should be same mds algorithm as matlab
    elif method == 'mds':
        mds = pcoa(data, number_of_dimensions=n_dim)
        transformed = mds.samples
        return transformed
    elif method == 'tsne':
        tsne = TSNE(n_components=n_dim)
        transformed = tsne.fit_transform(data)
        return transformed
        