"""
TBD @DANIEL
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, \
    Action, Namespace
from json import dumps
from os import getenv
from pathlib import Path
from shutil import rmtree
from typing import Optional
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
from numpy import corrcoef, save, array, median, unique

from MLDSP_core.__constants__ import methods_list
from MLDSP_core.classification import classify_dismat, calcInterclustDist
from MLDSP_core.one_dimensional_num_mapping import \
    one_dimensional_num_mapping_wrapper
from MLDSP_core.preprocessing import preprocessing
from MLDSP_core.visualisation import plotCGR, plot3d, \
    displayConfusionMatrix


def startCalcProcess(arguments: Namespace) -> Optional[str]:
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
    try:
        results_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("This output directory already exists, consider changing"\
              " the directory or Run name\n")
    method_num = arguments.method
    k_val = arguments.kmer
    method = methods_list[method_num]
    compute = one_dimensional_num_mapping_wrapper if method_num <= 13 \
        else method
    seq_dict, total_seq, cluster_dict, cluster_stats = preprocessing(
        data_set, metadata)
    q_names = None
    names, labels, seqs_length = zip(*[(a, cluster_dict[a], len(b))
                                       for a, b in seq_dict.items()])
    med_len = median(seqs_length)
    if args.query_seq_path is not None:
        q_results_path = results_path.joinpath('Testing')
        q_results_path.mkdir(parents=True)
        print(f'Query data path set to be: {args.query_seq_path}\n'
              f'Preprocessing')
        q_seqs, q_nseq, _, _ = preprocessing(args.query_seq_path, None,
                                             prefix='Query')
        q_names = list(q_seqs.keys())
        q_seqs_len = [len(b) for b in q_seqs.values()]
        q_med_len = median(q_seqs_len)
        q_log = open(results_path.joinpath('Query_run_data.txt'), 'x')
        q_log.write(f'Run_name: {run_name}(same as training run)\n'
        f'Method: {method}\nkmer:{k_val}\n'
        f'Median seq length: {q_med_len}\n'
        f'Training set median seq length:{med_len} NOTE: for one \
        dimensional numerical representations query seqs are normalized \
        to training set median seq length' f'Dataset size:{q_nseq}\n')

    log = open(results_path.joinpath('Run_data.txt'), 'x')
    log.write(f'Run_name: {run_name}\nMethod: {method}\nkmer: '
                f'{k_val}\nMedian seq length: {med_len}\nDataset size:'
                f'{total_seq}\nClass sizes:{cluster_stats}')

    print('Generating numerical sequences, applying DFT, computing '
          'magnitude spectra .... \n')

    parallel_results = Parallel(n_jobs=arguments.cpus)(delayed(compute)(
        seq=str(seq_dict[name]), name=name, results=results_path,
        order=arguments.order, kmer=k_val, med_len=med_len,
        method=method) for name in names)

    abs_fft_output, fft_output, cgr_output = zip(*parallel_results)
    q_distance_matx = None
    if q_names is not None:
        print('\tProcessing query')
        query_results = Parallel(n_jobs=arguments.cpus)(delayed(compute)(
            seq=str(q_seqs[q_name]), name=q_name, results=q_results_path,
            order=arguments.order, kmer=k_val, med_len=med_len,
            method=method, queryname='Query') for q_name in q_names)
        q_abs_fft_output, q_fft_output, q_cgr_output = zip(
            *query_results)
        #Build full distmat including train & query seqs
        full_matrix = (1 - corrcoef(abs_fft_output,q_abs_fft_output)) / 2
        #Subset matrix taking only query sample rows & trimming to
        #length to match training feature vector size
        q_distance_matx = full_matrix[total_seq:,:total_seq]
        del(full_matrix)
        q_out_fn = q_results_path.joinpath('dist_mat_query.npy').resolve()

    print('Building distance matrix')
    distance_matrix = (1 - corrcoef(abs_fft_output)) / 2
    out_fn = results_path.joinpath('dist_mat_train.npy').resolve()
    if not args.to_json:
        save(str(out_fn), distance_matrix)
        if args.query_seq_path is not None:
            save(str(q_out_fn), q_distance_matx)

    print('Performing classification .... \n')

    folds = arguments.folds if total_seq > arguments.folds else total_seq
    mean_accuracy, accuracy, cmatrix, mis_classified_dict, \
    full_model = classify_dismat(distance_matrix, array(labels),
                                 folds, cpus=arguments.cpus)
    cm_labels = unique(labels).tolist()
    q_preds = None
    if q_names is not None:
        print('Running query on trained models')
        q_preds = {model_name: model.predict(q_distance_matx)
                   for model_name, model in full_model.items()}
        q_log.write(str(q_preds))
        q_log.close()

    print('Scaling & data visualisation...')
    viz_path = results_path.joinpath('Images').resolve()
    if not args.to_json:
        viz_path.mkdir(parents=True, exist_ok=True)
        model_path = open(results_path.joinpath('Trained models.pkl'),'wb')
        pickle.dump(full_model,model_path)
        model_path.close()
    # CGR plotting
    if method_num == 14 or method_num == 15:
        for value in set (labels):
            index = labels.index(value)
            out = viz_path.joinpath(f'CGR {value}.png')
            cgr_img_data = plotCGR(cgr_output, sample_id=index,
                                out=out, to_json=args.to_json)
    else:
        cgr_img_data = None

    # 3d ModMap plotting
    mds = viz_path.joinpath('MoDmap.png')
    mds_img_data = plot3d(distance_matrix, labels, out=mds,
                          dim_res_method=args.dim_reduction,
                          to_json=args.to_json)
    cm_display_objs = displayConfusionMatrix(cmatrix, cm_labels)
    # Get intercluster distances
    intercluster_dist = calcInterclustDist(
        distance_matrix, labels)
    if not args.to_json:
        intercluster_dist.to_csv(viz_path.joinpath('Intercluster distance.csv'))
        for model, conf_mat in cm_display_objs.items():
            fig = conf_mat.figure_
            fig.savefig(viz_path.joinpath(f'Confusion matrix {model}'))
    outline = f'\n10X cross validation classifier accuracies:\n'
    outline += "\n".join([f"\t{m}: {ac}" for m, ac in accuracy.items()])
    log.write(outline)
    log.close()

    print('**** Processing completed ****\n')
    if args.to_json:
        return dumps({'mds': mds_img_data, 'cgr': cgr_img_data,
                      'cMatrix': cmatrix, 'cMatrixLabels': cm_labels,
                      'cMatrix_plots': cm_display_objs,
                      'modelAcc': accuracy,
                      'interclusterDist': intercluster_dist.to_html(),
                      'queryPredictions': q_preds})


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
    opt.add_argument('--dim_reduction', '-i', default='mds',
                     choices=['pca', 'mds', 'tsne'],
                     help='Type of dimensionality reduction technique to'
                          ' use in the visualization of the distance '
                          'matrix.')

    args = opt.parse_args()

    startCalcProcess(args)
