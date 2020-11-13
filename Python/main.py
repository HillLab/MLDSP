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
from functools import partial
import pywt
import os
from statistics import median, mean
import one_dimensional_num_mapping
## plotting
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# set up
if os.path.exists("Sequence_database.idx"):
  os.remove("Sequence_database.idx")
data_set = '../DataBase/Primates'
test_set = 'NoData'
seq_to_test=0
min_seq_len = 0
max_seq_len = 0
frags_per_seq = 1
methods_list = ['CGR(ChaosGameRepresentation)','Purine-Pyrimidine','Integer','Integer (other variant)','Real','Doublet','Codons','Atomic','EIIP','PairedNumeric','JustA','JustC','JustG','JustT','PuPyCGR','1DPuPyCGR']
method_num=0; # change method number referring the variable above (between 0 and 15)
k_val = 6; # used only for CGR-based representations(if methodNum=1,15,16)
# abs_fft_output_list=[]

seqs, cluster_names, number_of_clusters, cluster_samples_info, total_seq, cluster_dict = preprocessing(data_set)

print('Generating numerical sequences, applying DFT, computing magnitude spectra .... \n')
# variable holding all the keys (accession numbers) for corresponding clusters

keys = list(cluster_dict.keys())
values = list(cluster_dict.values())
# Could be parallelized in the future
seqs_length = [len(seqs[keys[i]].seq) for i in range(total_seq)]
med_len = median(seqs_length)

labels = [cluster_dict[x] for x in seqs.keys()]

## not needed for pool.map implementation
cgr_output_list = [[] for i in range(len(keys))]
fft_output_list = [[] for i in range(len(keys))]
abs_fft_output_list = [[] for i in range(len(keys))]



def compute_method_10_14(seq_index):
    # seqs is the sequence database, it is being called by accession number
    # (keys) which is being iterated over all seq_index,
    # (count of all sequences in uploaded dataset) .seq calls the string
    seq_new = str(seqs[keys[seq_index]].seq)
    if method_num==14:
        type(seq_new)
        seq_new = seq_new.replace('G', 'A')
        seq_new = seq_new.replace('C','T')
    cgr_output = cgr(seq_new,'ACGT',k_val)  # shape:[2^k, 2^k]
    cgr_output_list[seq_index] = cgr_output
    fft_output = fft.fft(cgr_output)  # shape:[2^k, 2^k]
    fft_output_list[seq_index] = fft_output
    #abs_fft_output_list[seq_index] = np.abs(fft_output.flatten())
    return np.abs(fft_output.flatten())  # flatted into 1d array & take absolute value of magnitude spectra


def compute_method_15(seq_index):
    seq_new = str(seqs[keys[seq_index]].seq)
    # creates PuPyCGR
    seq_new = seq_new.replace('G', 'A')
    seq_new = seq_new.replace('C', 'T')
    cgr_output= cgr(seq_new, 'ACGT', k_val)
    # takes only the last (bottom) row but all columns of cgr to make 1DPuPyCGR
    cgr_output = cgr_output[-1, :]
    cgr_output_list[seq_index] = cgr_output
    fft_output = fft.fft(cgr_output)  # shape:[2^k, 2^k] # this should be np.transpose(fft.fft(np.transpose(cgr_output)))
    fft_output_list[seq_index] = fft_output
    return(np.abs(fft_output.flatten()))  # flatted into 1d array

def one_dimensional_num_mapping_wrapper(seq_index):
    # normalize sequences to median sequence length of cluster
    seq_new = str(seqs[keys[seq_index]].seq)
    len(seq_new)
    if len(seq_new) >= med_len:
        seq_new = seq_new[1:round(med_len)]
    if method_num == 1:
        num_seq = one_dimensional_num_mapping.numMappingPP(seq_new)
    if len(num_seq) < med_len:
        pad_width = int(med_len - len(num_seq))
        num_seq = pywt.pad(num_seq,pad_width,'antisymmetric')[pad_width:]
    fft_output = fft.fft(num_seq)
    fft_output_list[seq_index] = fft_output
    return(np.abs(fft_output.flatten()))
# this can be replaced by Biopython
def compute_pearson_coeffient(x, y):
    r = pearsonr(x, y)[0]
    normalized_r = (1-r)/2
    return normalized_r


def compute_pearson_coeffient_wrapper(i,j,abs_fft_output_list):
    # print(abs_fft_output_list)
    # x = abs_fft_output_list[i]
    # y = abs_fft_output_list[indices[1]]
    # print(x)
    # print(y)
    return compute_pearson_coeffient(abs_fft_output_list[i], abs_fft_output_list[j])


# def phylogenetic_tree(distance_matrix):
#     constructor = DistanceTreeConstructor()
#     nj_tree = constructor.nj(distance_matrix)
#     # newick may not be need to be quoted
#     neighbour_joining_tree = nj_tree.format('newick')
#     upgma = constructor.upgma(distance_matrix)
#     upgma_tree = upgma.format('newick')
#     # can add code here for visualization with matplotlib


# This needs to be modified for compute canada parallel processing


pool = Pool(6)
if method_num == 0 or method_num == 14:
    abs_fft_output_list = pool.map(compute_method_10_14, range(total_seq))
elif method_num == 15:
    abs_fft_output_list = pool.map(compute_method_15, range(total_seq))
else:
    abs_fft_output_list = pool.map(one_dimensional_num_mapping_wrapper, range(total_seq))


    # seq_list = []
    # for seq_index in range(total_seq):
    #     seq_list.append(str(seqs[keys[seq_index]].seq))
    # fft_output_list, abs_fft_output_list, cgr_output_list = one_dim_num_rep_mapping(seq_list, method_num, med_len, total_seq)



distance_matrix = np.zeros(shape=(total_seq, total_seq))
distance_matrix = pool.starmap(partial(compute_pearson_coeffient_wrapper, abs_fft_output_list = abs_fft_output_list), ((i, j) for i in range(total_seq) for j in range(total_seq)))
distance_matrix = np.array(distance_matrix).reshape(total_seq, total_seq)

np.savetxt('distmat.txt', distance_matrix)


# Multi-dimensional Scaling: TODO

scaled_data = mds('distmat.txt')

# 3D  plot: TODO

# Classification

print('Performing classification .... \n');
folds=10
if (total_seq<folds):
    folds = totalSeq
accuracy = classify_dismat(distance_matrix,labels, folds, total_seq);
# accuracy,avg_accuracy, clNames, cMat
# accuracies = [accuracy, avg_accuracy];
print(accuracy)
print('**** Processing completed ****\n');
