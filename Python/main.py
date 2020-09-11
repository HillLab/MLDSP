from scipy import fft
import numpy as np
from multiprocessing import Pool
from scipy.stats import pearsonr
import Bio
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from classification import classify_dismat
from num_mapping import *
from preprocessing import *
from helpers import *

# set up
data_set = '/Users/dolteanu/local_documents/MATLAB/DataBase/Primates'
test_set = 'NoData'
seq_to_test=0
min_seq_len = 0
max_seq_len = 0
frags_per_seq = 1
methods_list = ['CGR(ChaosGameRepresentation)','Purine-Pyrimidine','Integer','Integer (other variant)','Real','Doublet','Codons','Atomic','EIIP','PairedNumeric','JustA','JustC','JustG','JustT','PuPyCGR','1DPuPyCGR']
methods_list[16]
method_num=0; # change method number referring the variable above (between 0 and 15)
k_val = 6; # used only for CGR-based representations(if methodNum=1,15,16)


# preprocess data
preprocessing(data_set)
# not needed since python has builtins & were loading seqs progressively
#max_len, min_len, mean_len, med_len = 0,0,0,0 # TODO: lengthCalc impl

fprintf('Generating numerical sequences, applying DFT, computing magnitude spectra .... \n');

cgr_output_list = []
fft_output_list = []
abs_fft_output_list = []

def compute_method_10_14(seq_index):
    seq_new = upper(seq[seq_index]);
    if method_num==15:
        seq_new = seq_new.replace('G', 'A')
        seq_new = seq_new.replacce('C','T');
    cgr_output = cgr(seq_new,'ACGT',k_val) # shape:[2^k, 2^k]
    cgr_output_list.append(cgr_output)
    fft_output = fft(cgr_output) # shape:[2^k, 2^k]
    fft_output_list.append(fft_output)
    abs_fft_output_list.append(np.abs(fft_output.flatten())) # flatted into 1d array


def compute_method_15(seq_index):
    sq_new = upper(seq(seq_index))
    seq_new = seq_new.replacce('G', 'A')
    seq_new = seq_new.replacce('C', 'T')
    cgr_output= cgr(seq_new, 'ACGT', k_val) # TODO: line 100 in completeScriptT.m does nothing?
    cgr_output_list.append(cgr_output)
    fft_output = fft(cgr_output) # shape:[2^k, 2^k]
    fft_output_list.append(fft_output)
    abs_fft_output_list.append(np.abs(fft_output.flatten())) # flatted into 1d array


def compute_pearson_coeffient(x, y):
    return pearsonr(x, y)


def compute_pearson_coeffient_wrapper(indices):
    compute_pearson_coeffient(*indices)


pool = Pool()
if method_num == 1 or method_num == 15:
    pool.map(compute_method_11_15, range(total_seq))
    # TODO: don't understand what reshape on line 85 mean
    # shape: [number of points, ((2^k)^2)]
elif method_num == 16:
    cgr_len = 2**k_val
    pool.map(compute_method_16, range(total_seq))
else:
    fft_output_list, abs_fft_output_list, cgr_output_list = oneDnumRepMethods(seq, method_num, med_len, total_seq) # TODO: oneDumRepMethods impl
# compute pearson correlation coefficient using parallel programming
distance_matrix = np.zeros(shape=(total_seq, total_seq))
for i in range(total_seq):
    if i == j:
        distance_matrix[i,j] = 1
    else:
        distance_matrix[i,j] = pool.map(compute_pearson_coeffient_wrapper, [(i, j) for j in range(total_seq)])


# Multi-dimensional Scaling: TODO
# 3D  plot: TODO
# Phylogenetic Tree: TODO # not tested

def phylogenetic_tree(distance_matrix):
    constructor = DistanceTreeConstructor()
    nj_tree = constructor.nj(distance_matrix)
    # newick may not be need to be quoted
    neighbour_joining_tree = nj_tree.format('newick')
    upgma = constructor.upgma(distance_matrix)
    upgma_tree = upgma.format('newick')
    # can add code here for visualization with matplotlib

# Classification
labels =[]
for i in range(number_of_clusters):
    for j in range(pointsPerCluster[i]):
        labels.append(i)

print('Performing classification .... \n');
folds=10
if (total_seq<folds):
    folds = totalSeq
[accuracy, avg_accuracy, clNames, cMat] = classify_dismat(distance_matrix,labels, folds, total_seq);
accuracies = [accuracy, avg_accuracy];

fprintf('**** Processing completed ****\n');
