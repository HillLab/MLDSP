from scipy import fft
import numpy as np
from multiprocessing import Pool
from scipy.stats import pearsonr
import Bio
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from classification import classify_dismat
from num_mapping import *
from preprocessing import preprocessing
from helpers import *
from cgr import *
# set up
data_set = '/Users/wanxinli/Desktop/project/MLDSP-desktop/samples/Primates'
test_set = 'NoData'
seq_to_test=0
min_seq_len = 0
max_seq_len = 0
frags_per_seq = 1
methods_list = ['CGR(ChaosGameRepresentation)','Purine-Pyrimidine','Integer','Integer (other variant)','Real','Doublet','Codons','Atomic','EIIP','PairedNumeric','JustA','JustC','JustG','JustT','PuPyCGR','1DPuPyCGR']
method_num=14; # change method number referring the variable above (between 0 and 15)
k_val = 6; # used only for CGR-based representations(if methodNum=1,15,16)


# there's probably a more appropriate way to import variables from a module
seqs, cluster_names, number_of_clusters, cluster_sample_count, total_seq, cluster_dict = preprocessing(data_set)
#max_len, min_len, mean_len, med_len = 0,0,0,0 # TODO: lengthCalc impl

print('Generating numerical sequences, applying DFT, computing magnitude spectra .... \n')

keys = list(cluster_dict.keys())

cgr_output_list = [[] for i in range(len(keys))]
fft_output_list = [[] for i in range(len(keys))]
abs_fft_output_list = [[] for i in range(len(keys))]


#seq_new = str(seqs[keys[1]].seq)
def compute_method_10_14(seq_index):
    seq_new = str(seqs[keys[seq_index]].seq)
    if method_num==14:
        seq_new = seq_new.replace('G', 'A')
        seq_new = seq_new.replace('C','T')
    cgr_output = cgr(seq_new,'ACGT',k_val) # shape:[2^k, 2^k]
    cgr_output_list[seq_index] = cgr_output
    fft_output = fft.fft(cgr_output) # shape:[2^k, 2^k]
    fft_output_list[seq_index] = fft_output
    abs_fft_output_list[seq_index] = np.abs(fft_output.flatten()) # flatted into 1d array
#compute_method_10_14(1)



def compute_method_15(seq_index):
    seq_new = str(seqs[keys[seq_index]].seq)
    seq_new = seq_new.replace('G', 'A')
    seq_new = seq_new.replace('C', 'T')
    cgr_output= cgr(seq_new, 'ACGT', k_val) # TODO: line 100 in completeScriptT.m does nothing?
    cgr_output_list[seq_index] = cgr_output
    fft_output = fft.fft(cgr_output) # shape:[2^k, 2^k]
    fft_output_list[seq_index] = fft_output
    abs_fft_output_list[seq_index] = np.abs(fft_output.flatten()) # flatted into 1d array
#compute_method_15(1)


def compute_pearson_coeffient(x, y):
    if x == y:
        return 1
    else:
        return pearsonr(x, y)


def compute_pearson_coeffient_wrapper(indices):
    compute_pearson_coeffient(*indices)


pool = Pool()
if method_num == 0 or method_num == 14:
    pool.map(compute_method_10_14, range(total_seq))
    # TODO: don't understand what reshape on line 85 mean
    # shape: [number of points, ((2^k)^2)]
elif method_num == 15:
    cgr_len = 2**k_val
    pool.map(compute_method_16, range(total_seq))
else:
    fft_output_list, abs_fft_output_list, cgr_output_list = oneDnumRepMethods(seq, method_num, med_len, total_seq) # TODO: oneDumRepMethods impl

print(len(abs_fft_output_list))
# compute pearson correlation coefficient using parallel programming
distance_matrix = np.zeros(shape=(total_seq, total_seq))
for i in range(total_seq):
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
