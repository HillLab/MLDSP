from spicy import fft
import numpy as np
from multiprocessing import Pool
from spicy.stats import pearsonr
import Bio
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
# set up
dataSet = 'C:\Users\GURJIT\Downloads\BcereusGroup\BcereusGroup'
testingSet = 'NoData'
seqToTest=0
minSeqLen = 0
maxSeqLen = 0
fragsPerSeq = 1
methods_list = {'CGR(ChaosGameRepresentation)','Purine-Pyrimidine','Integer','Integer (other variant)','Real','Doublet','Codons','Atomic','EIIP','PairedNumeric','JustA','JustC','JustG','JustT','PuPyCGR','1DPuPyCGR'};
method_num=1; # change method number referring the variable above (between 1 and 16)
k_val = 6; # used only for CGR-based representations(if methodNum=1,15,16)


# preprocess data
seq, ac_nmb, points_per_cluster, total_seq = 0,0,0,0 # TODO: assume from preprocess
max_len, min_len, mean_len, med_len = 0,0,0,0 # TODO: lengthCalc impl

fprintf('Generating numerical sequences, applying DFT, computing magnitude spectra .... \n');

cgr_output_list = []
fft_output_list = []
abs_fft_output_list = []

def compute_method_11_15(seq_index):
    seq_new = upper(seq[seq_index]);
    if method_num==15:
        seq_new = seq_new.replace('G', 'A')
        seq_new = seq_new.replacce('C','T');
    cgr_output = cgr(seq_new,'ACGT',k_val) # shape:[2^k, 2^k]
    cgr_output_list.append(cgr_output)
    fft_output = fft(cgr_output) # shape:[2^k, 2^k]
    fft_output_list.append(fft_output)
    abs_fft_output_list.append(np.abs(fft_output.flatten())) # flatted into 1d array


def compute_method_16(seq_index):
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
    pool.map(compute_method_11_15, total_seq)
    # TODO: don't understand what reshape on line 85 mean
    # shape: [number of points, ((2^k)^2)]
elif method_num == 16:
    cgr_len = 2**k_val
    pool.map(compute_method_16, total_seq)
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
[accuracy, avg_accuracy, clNames, cMat] = classificationCode(distance_matrix,labels, folds, total_seq);
accuracies = [accuracy, avg_accuracy];

fprintf('**** Processing completed ****\n');
