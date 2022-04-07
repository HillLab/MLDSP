"""
TBD @DANIEL
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, \
    Action
from functools import partial
from json import dumps
from os import getenv
from pathlib import Path
from typing import Optional, Union

from dill import load, dump
from joblib import Parallel, delayed, cpu_count
from numpy import corrcoef, save, array, median, unique

from MLDSP_core.__constants__ import methods_list, DEFAULT_METHOD, CGRS
from MLDSP_core.classification import classify_dismat, calcInterclustDist
from MLDSP_core.one_dimensional_num_mapping import \
    one_dimensional_num_mapping_wrapper
from MLDSP_core.preprocessing import preprocessing
from MLDSP_core.utils import Logger
from MLDSP_core.visualisation import plotCGR, plot3d, \
    displayConfusionMatrix


def startCalcProcess_test(query_seq_path: Path, run_name: str,
                          output_directory: Union[Path, str],
                          method: str = DEFAULT_METHOD,
                          kmer: int = 7, cpus: int = cpu_count(),
                          order: str = 'ACGT', to_json: bool = True,
                          **kwargs) -> Optional[str]:
    """
    Function to compute the prediction on a set of query sequences
    @DANIEL
    Args:
        query_seq_path:
        run_name:
        output_directory:
        method:
        kmer:
        cpus:
        order:
        to_json:
        **kwargs:

    Returns:

    """
    output_directory = Path(output_directory).resolve()
    results_path = output_directory.joinpath('Results', run_name).resolve()
    corr_fn = results_path.joinpath(f'{run_name}_partialcorr.pckl')
    full_model_path = results_path.joinpath('Trained_models.pkl')
    if not corr_fn.exists() or not full_model_path.exists():
        raise Exception('No training data in your path! please train the'
                        ' model first')
    with open(corr_fn, 'rb') as pick, open(full_model_path, 'rb') as fm:
        corr = load(pick)
        full_model = load(fm)

    q_results_path = results_path.joinpath('Testing')
    q_results_path.mkdir(parents=True, exist_ok=True)
    print(f'Query data path set to be: {query_seq_path}\nPreprocessing')
    compute = methods_list[method] if method in CGRS else \
        one_dimensional_num_mapping_wrapper
    q_seqs, q_nseq, _, _ = preprocessing(query_seq_path, None, prefix='Query')
    q_seqs_len = [len(b) for b in q_seqs.values()]
    q_med_len = median(q_seqs_len)
    q_log = Logger(q_results_path, f'Query_run_{run_name}.log')
    q_log.write(f'Run_name: {run_name}(same as training run)\n'
                f'Method: {method}\nkmer:{kmer}\n'
                f'Median seq length: {q_med_len}\n'
                f'NOTE: for one dimensional numerical representations '
                f'query seqs are normalized to training set median seq '
                f'length Dataset size:{q_nseq}\n')
    print('\tProcessing query')
    query_results = Parallel(n_jobs=cpus)(delayed(compute)(
        seq=str(q_seqs[q_name]), name=q_name, results=q_results_path,
        order=order, kmer=kmer, med_len=q_med_len, method=methods_list[method],
        queryname='Query') for q_name in q_seqs.keys())
    q_abs_fft_output, q_fft_output, q_cgr_output, _ = zip(*query_results)
    # Build full distmat including train & query seqs
    full_matrix = (1 - corr(y=q_abs_fft_output)) / 2
    # Subset matrix taking only query sample rows & trimming to
    # length to match training feature vector size
    total_seq = len(q_abs_fft_output)
    q_distance_matx = full_matrix[total_seq:, :-total_seq]
    del full_matrix
    q_out_fn = q_results_path.joinpath('dist_mat_query.npy').resolve()
    save(str(q_out_fn), q_distance_matx)
    print('Running query on trained models')
    q_preds = {model_name: model.predict(q_distance_matx)
               for model_name, model in full_model.items()}
    q_log.write(str(q_preds))
    if to_json:
        return dumps({'queryPredictions': q_preds})


def startCalcProcess_train(train_set: Path, train_labels: Union[Path, str],
                           run_name: str, output_directory: Union[Path, str],
                           method: str = DEFAULT_METHOD, kmer: int = 7,
                           folds: int = 10, cpus: int = cpu_count(),
                           order: str = 'ACGT', to_json: bool = True,
                           dim_reduction: str = 'mds', **kwargs
                           ) -> Optional[str]:
    """
    Train the models and run a 10 fold crossvalidation on the training data
    @Daniel
    Args:
        train_set:
        train_labels:
        run_name:
        output_directory:
        method:
        kmer:
        folds:
        cpus:
        order:
        to_json:
        dim_reduction:

    Returns:

    """
    output_directory = Path(output_directory).resolve()
    results_path = output_directory.joinpath('Results', run_name).resolve()
    try:
        results_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("This output directory already exists, consider changing"
              " the directory or Run name\n")
    compute = methods_list[method] if method in CGRS else \
        one_dimensional_num_mapping_wrapper
    seq_dict, total_seq, cluster_dict, cluster_stats = preprocessing(
        train_set, train_labels)
    med_len = median([len(x) for x in seq_dict.values()])
    log = Logger(results_path, f'Training_Run_{run_name}.txt')
    log.write(f'Run_name: {run_name}\nMethod: {method}\nkmer: {kmer}'
              f'\nMedian seq length: {med_len}\nDataset size: '
              f'{total_seq}\nClass sizes:{cluster_stats}\n')

    print('Generating numerical sequences, applying DFT, computing '
          'magnitude spectra .... \n')

    parallel_results = Parallel(n_jobs=cpus)(
        delayed(compute)(seq=str(seq_dict[name]), name=name, kmer=kmer,
                         results=results_path, order=order, med_len=med_len,
                         method=methods_list[method], label=cluster_dict[name]
                         ) for name in seq_dict.keys()
    )

    abs_fft_output, fft_output, cgr_output, labels = zip(*parallel_results)
    print('Building distance matrix')
    corr_fun = partial(corrcoef, abs_fft_output)
    distance_matrix = (1 - corr_fun()) / 2
    corr_fn = results_path.joinpath(f'{run_name}_partialcorr.pckl')
    with open(corr_fn, 'wb') as pickling:
        dump(corr_fun, pickling)
    if not to_json:
        out_fn = results_path.joinpath('dist_mat_train.npy').resolve()
        save(str(out_fn), distance_matrix)

    print('Performing classification .... \n')

    folds = folds if total_seq > folds else total_seq
    mean_accuracy, accuracy, cmatrix, mis_classified_dict, \
    full_model = classify_dismat(distance_matrix, array(labels), folds,
                                 cpus=cpus)
    cm_labels = unique(labels).tolist()

    print('Scaling & data visualisation...')
    viz_path = results_path.joinpath('Images').resolve()
    with open(results_path.joinpath('Trained_models.pkl'), 'wb') as model_path:
        dump(full_model, model_path)
    if not to_json:
        viz_path.mkdir(parents=True, exist_ok=True)
    # CGR plotting
    cgr_img_data = None
    if method == 'Pu/Py CGR' or method == 'Chaos Game Representation (CGR)':
        for value in set(labels):
            index = labels.index(value)
            log.write(f'CGR {value}: cgr_k={kmer}_{list(seq_dict.keys())[index]}.npy')
            out = viz_path.joinpath(f'CGR {value}.png')
            cgr_img_data = plotCGR(cgr_output, sample_id=index,
                                   out=out, to_json=to_json)
    # 3d ModMap plotting
    mds = viz_path.joinpath('MoDmap.png')
    mds_img_data = plot3d(distance_matrix, labels, out=mds,
                          dim_res_method=dim_reduction,
                          to_json=to_json)
    cm_display_objs = displayConfusionMatrix(cmatrix, cm_labels)
    # Get intercluster distances
    intercluster_dist = calcInterclustDist(
        distance_matrix, labels)
    if not to_json:
        intercluster_dist.to_csv(viz_path.joinpath('Intercluster_distance.csv'))
        for model, conf_mat in cm_display_objs.items():
            fig = conf_mat.figure_
            fig.savefig(viz_path.joinpath(f'Confusion matrix {model}'))
    outline = f'\n10X cross validation classifier accuracies:\n'
    outline += "\n".join([f"\t{m}: {ac}" for m, ac in accuracy.items()])
    log.write(outline)

    print('**** Processing completed ****\n')
    if to_json:
        return dumps({'mds': mds_img_data, 'cgr': cgr_img_data,
                      'cMatrix': cmatrix, 'cMatrixLabels': cm_labels,
                      'cMatrix_plots': cm_display_objs,
                      'modelAcc': accuracy,
                      'interclusterDist': intercluster_dist.to_html(),
                      })


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


    class HandleSpaces(Action):
        """
        Class to set the action to store as string
        """

        def __call__(self, parser, namespace, values, option_string=None):
            if not values:
                parser.error("You need to provide a string names or "
                             "filename")
            if isinstance(values, list):
                p = ' '.join(values)
            elif isinstance(values, str):
                p = values
            else:
                raise Exception('Wrong Type')
            choices = list(methods_list.keys())
            if p not in choices:
                exception_line = f"error: argument --method/-m: invalid " \
                                 f"choice: '{p}' (choose from " \
                                 f"{', '.join(choices)})"
                raise Exception(exception_line)
            setattr(namespace, self.dest, p)


    opt = ArgumentParser(usage='%(prog)s data_set_path metadata [options]',
                         formatter_class=ArgumentDefaultsHelpFormatter)
    opt.add_argument('train_set', help='Path to data set to train the'
                                       ' models with', action=PathAction)
    opt.add_argument('train_labels', help='CSV with the mapping of labels '
                                          'and sequence names')
    opt.add_argument('--query_seq_path', '-q', action=PathAction,
                     default=None, help='Path to test set fasta(s)')
    opt.add_argument('--run_name', '-r', help='Name of the run',
                     default='Bacteria')
    opt.add_argument('--output_directory', '-o', default=Path(),
                     action=PathAction,
                     help='Path to the output directory')
    opt.add_argument('--cpus', '-c', help='Number of cpus to use',
                     default=cpu_count(), type=int)
    opt.add_argument('--order', '-d', default='ACGT', type=str,
                     help='Order of the nucleotides in CGR, must be'
                          'uppercase')
    opt.add_argument('--kmer', '-k', help='Kmer size', default=5, type=int)
    opt.add_argument('--method', '-m', default=DEFAULT_METHOD, nargs='+',
                     action=HandleSpaces,
                     help='Name of the method (see more in the documentation)')
    opt.add_argument('--folds', '-f', default=10, type=int,
                     help='Number of folds for cross-validation')
    opt.add_argument('--to_json', '-j', default=False, action='store_true',
                     help='Number of folds for cross-validation')
    opt.add_argument('--dim_reduction', '-i', default='mds',
                     choices=['pca', 'mds', 'tsne'],
                     help='Type of dimensionality reduction technique to'
                          ' use in the visualization of the distance '
                          'matrix.')

    args = opt.parse_args()
    train_out = startCalcProcess_train(**vars(args))
    if args.query_seq_path is not None:
        test_out = startCalcProcess_test(**vars(args))
    print(f"{'*' * 8}\n* DONE *\n{'*' * 8}")
