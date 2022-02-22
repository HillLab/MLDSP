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
from MLDSP_core.one_dimensional_num_mapping import \
    one_dimensional_num_mapping_wrapper
from MLDSP_core.preprocessing import preprocessing
from MLDSP_core.visualisation import dimReduction


def main(arguments: Namespace):
    """
    @Daniel
    Args:
        arguments:
    """
    data_set = arguments.training_set
    metadata = arguments.training_labels
    run_name = arguments.run_name
    results_path = arguments.output_directory.joinpath(
        'Results', run_name).resolve()
    if results_path.exists():
        rmtree(results_path)
    results_path.joinpath("Num_rep", 'abs_fft').resolve().mkdir(
        parents=True)
    method_num = arguments.method
    k_val = arguments.kmer
    method = methods_list[method_num]
    compute = one_dimensional_num_mapping_wrapper if method_num <= 13 \
        else method
    seq_dict, total_seq, cluster_dict, cluster_stats = preprocessing(
        data_set, metadata)
    names, labels, seqs_length = zip(*[(a, cluster_dict[a], len(b))
                                       for a, b in seq_dict.items()])
    out_fn = results_path.joinpath('labels').resolve()
    np.save(str(out_fn), np.array(labels))
    med_len = np.median(seqs_length)
    print(f'Mean seq length: {med_len}')
    with open(results_path.joinpath('Run_data.txt'), 'x') as log:
        log.write(f'Run_name: {run_name}\nMethod: {method}\nkmer: '
                  f'{k_val}\nMedian seq length: {med_len}\nCluster '
                  f'sizes:{cluster_stats}')

    print('Generating numerical sequences, applying DFT, computing '
          'magnitude spectra .... \n')

    parallel_results = Parallel(n_jobs=arguments.cpus)(delayed(compute)(
        seq=str(seq_dict[name]), name=name, results=results_path, order=arguments.order,
        kmer=k_val, med_len=med_len, method=method) for name in names)

    abs_fft_output, fft_output, cgr_output = zip(*parallel_results)

    if method_num == 14 or method_num == 15:
        # TODO print first CGR from each class
        plt.matshow(cgr_output[0], cmap=cm.gray_r)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(results_path.joinpath('cgr_0.png'))

    print('Building distance matrix')

    distance_matrix = (1 - np.corrcoef(abs_fft_output)) / 2
    out_fn = results_path.joinpath('dist_mat').resolve()
    np.save(str(out_fn), distance_matrix)

    print('Scaling & data visualisation')
    # TODO Include what is in webapp
    le = pe.LabelEncoder()
    le.fit(labels)
    labs = le.transform(labels)
    scaled_distance_matrix = dimReduction(distance_matrix, n_dim=3,
                                          method='pca')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scaled_distance_matrix[:, 0], scaled_distance_matrix[:, 1],
               scaled_distance_matrix[:, 2], c=labs)
    plt.savefig(results_path.joinpath('MoDmap.png').resolve())

    # Classification
    print('Performing classification .... \n')
    folds = arguments.folds if total_seq > arguments.folds else total_seq
    mean_accuracy, accuracy, cmatrix, mis_classified_dict, \
    best_model = classify_dismat(distance_matrix, np.array(labels),
                                 folds)
    with open(results_path.joinpath('Run_data.txt'), 'a') as out:
        outline = f'\n10X cross validation classifier accuracies:\n'
        outline += "\n".join([f"\t{m}: {ac}" for m, ac in accuracy.items()])
        out.write(outline)
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
    opt.add_argument('training_set', help='Path to data set to train the'
                                          ' models with', action=PathAction)
    opt.add_argument('training_labels', help='CSV with the mapping of '
                                             'labels and sequence names')
    opt.add_argument('--run_name', '-r', help='Name of the run',
                     default='Bacteria')
    opt.add_argument('--output_directory', '-o', default=Path(),
                     action=PathAction,
                     help='Path to the output directory')
    opt.add_argument('--cpus', '-c', help='Number of cpus to use',
                     default=cpu_count(), type=int)
    opt.add_argument('--order', '-d', default='ACGT', type=str,
                     help='Order of the nucleotides in CGR')
    opt.add_argument('--kmer', '-k', help='Kmer size', default=5, type=int)
    opt.add_argument('--method', '-m', default=14, type=int,
                     choices=range(1, 17),
                     help='Numeric representation of method (see more in'
                          ' the documentation)')
    opt.add_argument('--folds', '-f', default=10, type=int,
                     help='Number of folds for cross-validation')
    opt.add_argument('--to_json', '-j', default=False, action='store_true',
                     help='Number of folds for cross-validation')

    args = opt.parse_args()

    main(args)
