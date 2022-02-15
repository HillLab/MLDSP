"""
TBD @DANIEL
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, \
    Action
from functools import partial
from multiprocessing import Pool, cpu_count
from os import getenv
from pathlib import Path
from shutil import rmtree
from typing import Dict, Callable, Tuple, Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pyfaidx import FastaRecord
from pywt import pad
from scipy import fft
from sklearn import preprocessing as pe

from MLDSP_core.cgr import cgr
from MLDSP_core.classification import classify_dismat
from MLDSP_core.one_dimensional_num_mapping import *
from MLDSP_core.preprocessing import preprocessing
from MLDSP_core.visualisation import dimReduction


def compute_cgr(seq: FastaRecord, method_number: int, kmer: int,
                results: Path, order: str = 'ACGT'
                ) -> Tuple[Any, Any, Any]:  # TODO change any for the actual signature
    """
    This function compute the CGR matrix for a sequence in seq_dict
    Args:
        seq: FastaRecord instance with the sequence and name of the sequence
        method_number: Method used to compute CGR
        kmer: Kmer value to use
        results: Path to the results folder
        order: Order of the nucleotides in the Chaos square

    Returns:

    """
    seq_new = str(seq)
    name = seq.name
    # Replace complementary Purine/Pyrimidine
    if method_number == 15 or method_number == 16:
        seq_new = seq_new.replace('G', 'A').replace('C', 'T')
    cgr_raw = cgr(seq_new, order, kmer)
    # takes only the last (bottom) row but all columns of cgr to make 1DPuPyCGR
    if method_number == 16:
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


def one_dimensional_num_mapping_wrapper(
        seq_index: int, method: Callable, seq_dict: Dict[str], keys: str,
        med_len: int, results_path: Path):
    """
    TBD @Daniel
    Args:
        seq_index:
        method:
        seq_dict:
        keys:
        med_len:
        results_path:

    Returns:

    """
    # normalize sequences to median seq length of cluster
    seq_new = str(seq_dict[keys[seq_index]].seq)
    if len(seq_new) >= med_len:
        seq_new = seq_new[0:round(med_len)]
    num_seq = method(seq_new)
    if len(num_seq) < med_len:
        pad_width = int(med_len - len(num_seq))
        num_seq = pad(num_seq, pad_width, 'antisymmetric')[pad_width:]
    ofname = results_path.joinpath('Num_rep',
                                   f'{method}_{seq_index}').resolve
    np.save(str(ofname), num_seq)
    fft_output = fft.fft(num_seq)
    abs_fft_output = np.abs(fft_output.flatten())
    return abs_fft_output, fft_output


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

    args = opt.parse_args()

    data_set = args.data_set
    metadata = args.metadata
    run_name = args.run_name
    results_path = args.output_directory.joinpath('Results', run_name
                                                  ).resolve()
    if results_path.exists():
        rmtree(results_path)
    results_path.joinpath("Num_rep", 'abs_fft').resolve().mkdir(
        parents=True)

    # Dictionary order & names dependent for downstream execution 
    methods_list = {1: num_mapping_PP,
                    2: num_mapping_Int,
                    3: num_mapping_IntN,
                    4: num_mapping_Real,
                    5: num_mapping_Doublet,
                    6: num_mapping_Codons,
                    7: num_mapping_Atomic,
                    8: num_mapping_EIIP,
                    9: num_mapping_AT_CG,
                    10: num_mapping_justA,
                    11: num_mapping_justC,
                    12: num_mapping_justG,
                    13: num_mapping_justT,
                    14: 'cgr',
                    15: 'PuPyCGR',
                    16: '1DPuPyCGR'}
    # Change method number referring the variable above (between 1 and 16)
    method_num = 14
    k_val = 5  # used only for CGR-based representations(if methodNum=14,15,16)
    # End of user set up

    max_clust_size = 10000000
    # frags_per_seq = 1
    method = methods_list.get(method_num)
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

    if method_num == 14 or method_num == 15 or method_num == 16:
        abs_fft_output, fft_output, cgr_output = zip(*Parallel(
            n_jobs=args.cpus)(delayed(compute_cgr)(
            sequence, method_num, k_val, results_path, args.order)
                              for name, sequence in seq_dict.items()))

    with Pool(args.cpus) as pool:
        if method_num == 14 or method_num == 15 or method_num == 16:
            abs_fft_output, fft_output, cgr_output = zip(*pool.map(partial(
                compute_cgr, seq_dict=seq_dict, keys=seq_dict.keys(),
                method_num=method_num, k_val=k_val, results_path=results_path
            ), range(total_seq)))
        else:
            abs_fft_output, fft_output = zip(*pool.map(partial(
                one_dimensional_num_mapping_wrapper, method=method,
                seq_dict=seq_dict, keys=seq_dict.keys(), med_len=med_len,
                result_path=results_path), range(total_seq)))

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
