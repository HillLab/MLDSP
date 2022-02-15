"""
TBD @DANIEL
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, \
    Action, Namespace
from os import getenv
from pathlib import Path
from shutil import rmtree

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, cpu_count
from sklearn import preprocessing as pe

from MLDSP_core.__constants__ import methods_list
from MLDSP_core.classification import classify_dismat
from MLDSP_core.one_dimensional_num_mapping import one_dimensional_num_mapping_wrapper
from MLDSP_core.preprocessing import preprocessing
from MLDSP_core.visualisation import dimReduction


def main(args: Namespace):
    data_set = args.data_set
    metadata = args.metadata
    run_name = args.run_name
    results_path = args.output_directory.joinpath('Results', run_name
                                                  ).resolve()
    if results_path.exists():
        rmtree(results_path)
    results_path.joinpath("Num_rep", 'abs_fft').resolve().mkdir(
        parents=True)

    # Change method number referring the variable above (between 1 and 16)
    method_num = args.method
    # used only for CGR-based representations(if methodNum=14,15,16)
    k_val = args.kmer
    # End of user set up

    max_clust_size = 10000000
    # frags_per_seq = 1
    method = methods_list[method_num]
    compute = one_dimensional_num_mapping_wrapper if method_num <= 13 else method
    seq_dict, total_seq, cluster_dict, cluster_stats = preprocessing(
        data_set, max_clust_size, metadata)
    labels, seqs_length = [], []
    applab, appsl = labels.append, seqs_length.append
    for accession, sequence in seq_dict.items():
        applab(cluster_dict[accession])
        appsl(len(sequence))
    out_fn = results_path.joinpath('labels').resolve()
    np.save(str(out_fn), np.array(labels))
    med_len = np.median(seqs_length)
    print(f'Mean seq length: {med_len}')
    seq_new_list = []
    seq_list = []
    with open(results_path.joinpath('Run_data.txt'), 'x') as log:
        log.write(f'Run_name: {run_name}\nMethod: {method}\nkmer: '
                  f'{k_val}\nMedian seq length: {med_len}\nCluster '
                  f'sizes:{cluster_stats}')

    print('Generating numerical sequences, applying DFT, computing '
          'magnitude spectra .... \n')

    abs_fft_output, fft_output, cgr_output = zip(*Parallel(n_jobs=args.cpus)(
        delayed(compute)(seq=sequence, results=results_path, order=args.order,
                         kmer=k_val, med_len=med_len,
                         method=method)
        for _, sequence in seq_dict.items()))

    if method_num == 14 or method_num == 15:
        plt.matshow(cgr_output[0], cmap=cm.gray_r)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(results_path.joinpath('cgr_0.png'))

    print('Building distance matrix')

    # # Numpy implementation
    distance_matrix = (1 - np.corrcoef(abs_fft_output)) / 2
    out_fn = results_path.joinpath('dist_mat').resolve()
    np.save(str(out_fn), distance_matrix)

    print('Building Phylogenetic Trees')

    print('Scaling & data visualisation')
    le = pe.LabelEncoder()
    le.fit(labels)
    labs = le.transform(labels)
    scaled_distance_matrix = dimReduction(distance_matrix, n_dim=3,
                                          method='pca')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scaled_distance_matrix[:, 0], scaled_distance_matrix[:, 1],
               scaled_distance_matrix[:, 2], c=labs)
    plt.savefig(results_path.joinpath('MoDmap.png').resolve)

    # Classification
    print('Performing classification .... \n')
    folds = 10
    if total_seq < folds:
        folds = total_seq
    mean_accuracy, accuracy, cmatrix, misClassifiedDict = classify_dismat(
        distance_matrix, labels, folds, total_seq)
    # accuracy,avg_accuracy, clNames, cMat
    # accuracies = [accuracy, avg_accuracy];
    print('\n10X cross validation classifier accuracies', accuracy,
          file=open(results_path.joinpath('Run_data.txt'), 'a'))
    print('**** Processing completed ****\n')


if __name__ == '__main__':
    class PathAction(Action):
        """
        Class to set the action to store as path
        """

        def __call__(self, parser, namespace, values, option_string=None):
            if not values:
                parser.error("You need to provide a string with path or "
                             "filename")
            p = Path(values).resolve()
            if not (p.is_file() or p.is_dir()):
                p = Path(getenv(values)).resolve()

            setattr(namespace, self.dest, p)


    opt = ArgumentParser(usage='%(prog)s data_set_path metadata [options]',
                         formatter_class=ArgumentDefaultsHelpFormatter)
    opt.add_argument('data_set', help='Path to data set to be analysed',
                     action=PathAction)
    opt.add_argument('metadata', help='Metadata of your sample sequences')
    opt.add_argument('--run_name', '-r', help='Name of the run',
                     default='Bacteria')
    opt.add_argument('--output_directory', '-o',
                     help='Path to the output directory',
                     default=Path(), action=PathAction)
    opt.add_argument('--cpus', '-c', help='Number of cpus to use',
                     default=cpu_count(), type=int)
    opt.add_argument('--order', '-d', help='Order of the nucleotides in CGR',
                     default='ACGT', type=str)
    opt.add_argument('--kmer', '-k', help='Kmer size', default=5, type=int)
    opt.add_argument('--method', '-m', default=14, type=int,
                     help='Numeric representation of method (see more in'
                          ' the documentation)')

    args = opt.parse_args()

    main(args)
