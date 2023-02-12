"""
Main script for execution of MLDSP backend computation

Functions:
    startCalcProcess_train(**vars(args)) -> Results/{run_name}
    startCalcProcess_test(**vars(args)) -> Results/{run_name}/Testing
    execute(command line arguments)-> startCalcProcess_train(**vars(args)), Optional[startCalcProcess_test(**vars(args))]
"""
import cProfile
profiler = cProfile.Profile()
profiler.enable()
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial
from json import dumps
from pathlib import Path
from sys import stdout
from typing import Optional, Union
from collections import Counter
from dill import load, dump
from joblib import Parallel, delayed, cpu_count
from numpy import corrcoef, save, array, median, unique, max, min

from MLDSP_core.__constants__ import methods_list, DEFAULT_METHOD, CGRS
from MLDSP_core.classification import classify_dismat, calcInterclustDist
from MLDSP_core.one_dimensional_num_mapping import \
    one_dimensional_num_mapping_wrapper
from MLDSP_core.preprocessing import preprocessing
from MLDSP_core.utils import Logger, uprint, PathAction, HandleSpaces
from MLDSP_core.visualisation import plotCGR, plot3d, \
    displayConfusionMatrix


def startCalcProcess_test(query_seq_path: Path, run_name: str,
                          output_directory: Union[Path, str],
                          training_med: float, method: str = DEFAULT_METHOD,
                          kmer: int = 7, cpus: int = cpu_count(),
                          order: str = 'ACGT', to_json: bool = True,
                          **kwargs) -> Optional[str]:
    """
    Function to compute the predicted class on a set of query
    sequences with no label from a previously trained model.
    Args:
        query_seq_path: Path to folder of fastas to be classified
        run_name: Unique name used to track datasets run
        output_directory: path to desired output location for results
        method: numerical representation method to be used, see args
                for available options
        kmer: size of k-length oligomers to be used for fCGR generation
        cpus: number of cpu cores to use for parallel computing
        order: assignment of fCGR vertex labels: 
               'bottom left, top left, top right, bottom right'
               Should not be modified unless using non-standard
               fCGR or input sequences is not nucleotides.
        to_json: Boolean to specify command-line output rather
                 than GUI json output
        **kwargs: Other keyword arguments, corresponding to function 
                  variables, none defined

    Returns:
        Query_run_{run_name}.log: Text log file
        
    """
    output_directory = Path(output_directory).resolve()
    results_path = output_directory.joinpath('Results', run_name).resolve()
    # Path to corresponding distance matrix from training run, required for
    # calculating query feature vectors (input into classifiers)
    corr_fn = results_path.joinpath(f'{run_name}_partialcorr.pckl')
    # Path to pre-trained ML models
    full_model_path = results_path.joinpath('Trained_models.pkl')
    # Check if training data exists & is in right format
    if not corr_fn.exists() or not full_model_path.exists():
        raise Exception('No training data in your path! please train the'
                        ' model first')
    # Open correlation function/original dist mat & trained ML models
    with open(corr_fn, 'rb') as pick, open(full_model_path, 'rb') as fm:
        corr = load(pick)
        full_model = load(fm)
    # Set new paths for query results
    q_results_path = results_path.joinpath('Testing')
    q_results_path.mkdir(parents=True, exist_ok=True)
    print_file = stdout
    if 'quiet' in kwargs:
        if kwargs['quiet']:
            print_file = results_path.joinpath('prints.txt')
    uprint(f'Query data path set to be: {query_seq_path}\nPreprocessing...',
           print_file=print_file)
    # Select numerical representation
    compute = methods_list[method] if method in CGRS else \
        one_dimensional_num_mapping_wrapper
    # Read fasta & perform preprocessing
    q_seqs, q_nseq, _ = preprocessing(query_seq_path, None,
                                         prefix='Query',
                                         print_file=print_file,
                                         output_path=q_results_path)
    # Get dataset statistics & print
    q_seqs_len = [len(b) for b in q_seqs.values()]
    q_med_len = median(q_seqs_len)
    q_max_len = max(q_seqs_len)
    q_min_len = min(q_seqs_len)
    q_log = Logger(q_results_path, f'Query_run_{run_name}.log')
    q_log.write(f'Run_name: {run_name}(same as training run)\n'
                f'Method: {method}\nkmer:{kmer}\n'
                f'Median seq length: {q_med_len} Shortest sequence: {q_min_len} '
                f'Longest sequence: {q_max_len}\n'
                f'NOTE: for one dimensional numerical representations '
                f'query seqs are normalized to training set median seq '
                f'length:{training_med}\nDataset size:{q_nseq}\n')
    uprint('\tProcessing query\n', print_file=print_file)
    # Calculate num. representation, FFT and CGR in parallel
    query_results = Parallel(n_jobs=cpus)(delayed(compute)(
        seq=str(q_seqs[q_name]), name=q_name, results=q_results_path,
        order=order, kmer=kmer, med_len=training_med,
        method=methods_list[method], queryname='Query')
                                          for q_name in q_seqs.keys())
    q_abs_fft_output, q_fft_output, q_cgr_output, _ = zip(*query_results)
    # Build full distmat including train & query seqs
    full_matrix = (1 - corr(y=q_abs_fft_output)) / 2
    # Subset matrix taking only query sample rows & trimming to
    # length to match training feature vector size
    q_distance_matx = full_matrix[-q_nseq:, :-q_nseq]
    del full_matrix
    # Save query sample vectors, only useable with specific trained model
    # for reproducibility, fundamental problem with MLDSP 
    q_out_fn = q_results_path.joinpath('dist_mat_query.npy').resolve()
    save(str(q_out_fn), q_distance_matx)
    uprint('Predicting query labels with trained models\n',
           print_file=print_file)
    # ML model prediction
    q_preds = {model_name: model.predict(q_distance_matx)
               for model_name, model in full_model.items()}
    q_log.write(str(q_preds))
    if to_json:
        json_str = dumps({'queryPredictions': {k: v.tolist(
        ) for k, v in q_preds.items()}})
        uprint(json_str)
        return json_str


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
    # Results path and output setup
    output_directory = Path(output_directory).resolve()
    results_path = output_directory.joinpath('Results', run_name).resolve()
    print_file = stdout
    if 'quiet' in kwargs:
        if kwargs['quiet']:
            print_file = results_path.joinpath('prints.txt')
    # Select numerical representation
    compute = methods_list[method] if method in CGRS else \
        one_dimensional_num_mapping_wrapper
    uprint('Building Fasta index\n', print_file=print_file)
    # Read fasta & metadata, perform data checks
    seq_dict, total_seq, cluster_dict = preprocessing(
        train_set, train_labels, print_file=print_file,
        output_path=results_path)
    # Get dataset statistics & print
    seqs_len = [len(x) for x in seq_dict.values()]
    med_len = median(seqs_len)
    max_len = max(seqs_len)
    min_len = min(seqs_len)
    # More path setup
    corr_fn = results_path.joinpath(f'{run_name}_partialcorr.pckl')
    full_model_path = results_path.joinpath('Trained_models.pkl')
    class_report_path = results_path.joinpath((f'{run_name}_classification_report.pkl'))
    conf_matrix_path = results_path.joinpath((f'{run_name}_confusion_matrices.pkl'))
    mis_classified_path = results_path.joinpath((f'{run_name}_confusion_matrices.pkl'))
    # Checks if training with given run name has already been done; for query prediction
    if corr_fn.exists() and full_model_path.exists():
        uprint("Training with this name has already be run. Using stored"
               " values\n", print_file=print_file)
        return None, med_len
    try:
        results_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if any(results_path.glob('[!prints.txt]')):
            raise Exception("This output directory already exists, "
                            "consider changing the directory or Run name")
    log = Logger(results_path, f'Training_Run_{run_name}.log')
    log.write(f'Run_name: {run_name}\nMethod: {method}\nkmer: {kmer}'
              f'\nMedian seq length: {med_len} Shortest sequence: {min_len} '
              f'Longest sequence: {max_len}\n'
              f'Dataset size: {total_seq}\n')

    uprint('Generating numerical sequences, applying DFT, computing '
           'magnitude spectra .... \n', print_file=print_file)
    # Calculate num. representation, FFT and CGR in parallel
    parallel_results = Parallel(n_jobs=cpus)(
        delayed(compute)(seq=str(seq_dict[name]), name=name, kmer=kmer,
                         results=results_path, order=order, med_len=med_len,
                         method=methods_list[method], label=cluster_dict[name]
                         ) for name in seq_dict.keys()
    )

    abs_fft_output, fft_output, cgr_output, labels = zip(*parallel_results)
    log.write(f'Class sizes:\n')
    class_sizes = Counter(labels)
    [log.write(f'{key}:{value}, ') for (key, value) in class_sizes.items()]
    log.write(f'\n')
    uprint('Building distance matrix\n', print_file=print_file)
    # Partial function combining the Pearson correlation coefficient function
    # with the fft magnitude spectra, since no Y= is specified 
    # correlation coefficient is calculated pairwise. 
    # Required to later compute pairwise between original magnitude spectra and new 
    # query spectra, in test function: (Y=) to get same size of rows in distance matrix
    corr_fun = partial(corrcoef, abs_fft_output)
    # Actual distance matrix computation
    distance_matrix = (1 - corr_fun()) / 2
    # Save partial function for extending distance matrix (input feature vectors)
    # during query sample prediction
    with open(corr_fn, 'wb') as pickling:
        dump(corr_fun, pickling)
    if not to_json:
        out_fn = results_path.joinpath('dist_mat_train.npy').resolve()
        save(str(out_fn), distance_matrix)

    uprint('Performing classification .... \n', print_file=print_file)
    # ML classification
    smallest_class_count = class_sizes.most_common()[-1][1]
    folds = folds if smallest_class_count > folds else smallest_class_count
    mean_accuracy, accuracy, cmatrix, mis_classified_dict, \
    full_model, class_reports = classify_dismat(distance_matrix, array(labels), folds,
                                 log, cpus=cpus, print_file=print_file)
    # Set up for saving results
    cm_labels = unique(labels).tolist()
    with open(full_model_path, 'wb') as model_path:
        dump(full_model, model_path)
    with open(class_report_path, 'wb') as report_path:
        dump(class_reports, report_path)
    with open(conf_matrix_path, 'wb') as confusion_path:
        dump(cmatrix, confusion_path)
    with open(mis_classified_path, 'wb') as misclassified_path:
        dump(mis_classified_dict, misclassified_path)

    uprint('Scaling & data visualisation...\n', print_file=print_file)
    viz_path = results_path.joinpath('Images').resolve()
    if not to_json:
        viz_path.mkdir(parents=True, exist_ok=True)
    # CGR plotting
    if method == 'Pu/Py CGR' or method == 'Chaos Game Representation (CGR)':
        cgr_img_data = None
        out = viz_path.joinpath(f'CGRs {run_name}.svg')
        cgr_img_data = plotCGR(cgr_output, labels, seq_dict, log, kmer, out, to_json)
    # 3d ModMap plotting
    mds = viz_path.joinpath(f'MoDmap_{dim_reduction}.json')
    mds_img_data = plot3d(distance_matrix, labels, out=mds,
                          dim_res_method=dim_reduction,
                          to_json=to_json)
    # Confusion matrices plotting
    prefix_viz = viz_path.joinpath('Confusion_matrix')
    cm_display_objs = displayConfusionMatrix(cmatrix, cm_labels,
                                             prefix=prefix_viz,
                                             to_json=to_json)
    # Get intercluster distances
    intercluster_dist = calcInterclustDist(distance_matrix, labels)
    if not to_json:
        intercluster_dist.to_csv(viz_path.joinpath('Intercluster_distance.csv'),float_format="%.3f")
    # Print overall accuracies
    outline = f'\n10X cross validation classifier balanced accuracies:\n'
    outline += "\n".join([f"\t{m}: {ac}" for m, ac in accuracy.items()])
    log.write(outline)

    uprint('**** Processing completed ****\n', print_file=print_file)
    json_str = None
    if to_json:
        j_cmatrix = {k: v.tolist() for k, v in cmatrix.items()}
        json_str = dumps({'mds': mds_img_data, 'cgr': cgr_img_data,
                          'cMatrix': j_cmatrix, 'cMatrixLabels': cm_labels,
                          'cMatrix_plots': cm_display_objs,
                          'modelAcc': accuracy,
                          'interclusterDist': intercluster_dist.to_html()
                          })
        uprint(json_str)
    return json_str, med_len


def execute():
    opt = ArgumentParser(usage='%(prog)s data_set_path metadata [options]',
                         formatter_class=ArgumentDefaultsHelpFormatter)
    opt.add_argument('train_set', help='Path to data set to train the'
                                       ' models with', action=PathAction)
    opt.add_argument('train_labels', help='CSV with the mapping of labels '
                                          'and sequence names')
    opt.add_argument('--query_seq_path', '-q', action=PathAction,
                     default=None, help='Path to test set fasta(s)')
    opt.add_argument('--run_name', '-r', help='Name of the run',
                     default='Default')
    opt.add_argument('--output_directory', '-o', default=Path(),
                     action=PathAction,
                     help='Path to the output directory')
    opt.add_argument('--cpus', '-c', help='Number of cpus to use',
                     default=cpu_count(), type=int)
    opt.add_argument('--order', '-d', default='ACGT', type=str,
                     help='Order of the nucleotides in CGR, must be'
                          'uppercase')
    opt.add_argument('--kmer', '-k', help='Kmer size', default=7, type=int)
    opt.add_argument('--method', '-m', default=DEFAULT_METHOD, nargs='+',
                     action=HandleSpaces,
                     help='Name of the method (see more in the documentation)')
    opt.add_argument('--folds', '-f', default=10, type=int,
                     help='Number of folds for cross-validation')
    opt.add_argument('--to_json', '-j', default=False, action='store_true',
                     help='Boolean, to output results in json for API to'
                     'MLDSP-web frontend')
    opt.add_argument('--dim_reduction', '-i', default='mds',
                     choices=['pca', 'mds', 'tsne'],
                     help='Type of dimensionality reduction technique to'
                          ' use in the visualization of the distance '
                          'matrix.')
    opt.add_argument('--quiet', '-z', default=False, action='store_true',
                     help='Do not print anything to stdout but the json (if selected)')
    args = opt.parse_args()
    train_out, train_med_len = startCalcProcess_train(**vars(args))
    if args.query_seq_path is not None:
        test_out = startCalcProcess_test(training_med=train_med_len,
                                         **vars(args))
    if not args.quiet:
        print(f"{'*' * 8}\n* DONE *\n{'*' * 8}")
    profiler.disable()
    profiler.dump_stats(f'{args.run_name}_profile.prof')
if __name__ == '__main__':
    execute()
