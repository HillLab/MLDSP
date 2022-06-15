"""
@Daniel
"""
from base64 import b64encode
from io import BytesIO
from json import dumps
from pathlib import Path
from typing import Union, DefaultDict, List, Dict

from matplotlib import cm
from matplotlib import pyplot as plt
# from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame
from plotly import express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import ConfusionMatrixDisplay


def dimReduction(data: ndarray, n_dim: int, method: str) -> ndarray:
    """
    Function will take in a nxm 2d-array and reduce the dimensions of
    the data using a specified dimensionality reduction technique (PCA,
    MDS, or TSNE).
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
        mds = MDS(n_components=n_dim, dissimilarity='precomputed')
        transformed = mds.fit_transform(data)
        return transformed
    elif method == 'tsne':
        tsne = TSNE(n_components=n_dim)
        transformed = tsne.fit_transform(data)
        return transformed


def plotCGR(cgr_output: ndarray, labels: tuple, seq_dict,
            log, kmer:  int, out: Path = Path('CGR.png'),
            to_json: bool = False) -> Union[None, b64encode]:
    """
    returns matplotlib Figure object

    Args:
        sample_id: Index of the sample to render
        out: Path for output
        to_json: Dump figure to json
        cgr_output: array containing all raw cgrs

    Returns: cgrFig
    """
    extent = 0,1,0,1
    unique_classes = set(labels)
    cgrFig, axs = plt.subplots(1,len(unique_classes),figsize=(28, 7))
    counter = 0
    for value in unique_classes:
        subplot = axs[counter]
        index = labels.index(value)
        subplot.matshow(cgr_output[index],cmap=cm.gray_r, extent = extent)
        subplot.set_title(f'CGR {value} (k={kmer})')
        subplot.set_axis_off()
        subplot.text(-0.1,-0.1,"A",fontsize=15)
        subplot.text(1,-0.1,"T",fontsize=15)
        subplot.text(-0.1,1.1,"C",fontsize=15)
        subplot.text(1,1.1,"G",fontsize=15)
        log.write(f'CGR {value}: cgr_k={kmer}_{list(seq_dict.keys())[index]}.npy\n')
        counter += 1
    buf = BytesIO() if to_json else out
    cgrFig.savefig(buf, format="png")
    if to_json:
        cgrImgData = b64encode(buf.getbuffer()).decode("ascii")
        return cgrImgData


def plot3d(dist_matrix: ndarray, labels: list, out: Path = 'MDS.png',
           dim_res_method: str = 'mds', to_json: bool = False
           ) -> Union[None, dumps]:
    """
    @Daniel
    Args:
        dim_res_method: Type of dimensionality reduction to use
        out: path (with filename) for output (incase of to_json to be false)
        to_json: Dump figure to json
        dist_matrix:
        labels:

    Returns:

    """
    scaled_distance_matrix = dimReduction(dist_matrix, n_dim=3,
                                          method=dim_res_method)
    coordDf = DataFrame(scaled_distance_matrix, columns=['X', 'Y', 'Z'])
    labelsFormatted = [label + "    " for label in labels]
    coordDf['label'] = labelsFormatted
    fig = px.scatter_3d(coordDf, x='X', y='Y', z='Z', color='label',
                        opacity=0.9)
    # tight layout
    fig.update_layout(margin=dict(l=20, r=0, b=0, t=20),
                      legend_title_text="Classes")
    if to_json:
        # mdsGraphJSON = dumps(fig, cls=PlotlyJSONEncoder)
        # return mdsGraphJSON
        return fig.to_json()
    else:
        fig.write_image(out)


def displayConfusionMatrix(confMatrix: DefaultDict[str, ndarray],
                           alabels: List[str], prefix: Path = 'cm',
                           format: str = 'png', to_json: bool = False
                           ) -> Dict[str, ConfusionMatrixDisplay]:
    """
    @Daniel
    Args:
        format:
        to_json:
        confMatrix:
        alabels:

    Returns:
    """
    conf_matrix_display_objs = {}
    for model, matrix in confMatrix.items():
        filename = f'{prefix}_{model}.{format}'
        cmd = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                     display_labels=alabels)
        ax = cmd.plot(cmap='Blues', colorbar=False)
        fig = ax.figure_
        buf = BytesIO() if to_json else filename
        fig.savefig(buf, format="png")
        if to_json:
            conf_matrix_display_objs[model] = b64encode(buf.getbuffer()
                                                        ).decode("ascii")
    return conf_matrix_display_objs
