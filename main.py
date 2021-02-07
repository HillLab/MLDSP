import sys
from scipy import fft
import numpy as np
from multiprocessing import Pool
from scipy.stats import pearsonr
# import Bio
from Bio.Phylo.TreeConstruction import *
from functools import partial
import pywt
# import os
from statistics import median, mean
from one_dimensional_num_mapping import *
# plotting
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from classification import classify_dismat
from preprocessing import preprocessing
from helpers import *
from cgr import *
from visualisation import dimReduction
# user set up
data_set = './DataBase/Primates'
# Dictionary order & names dependent for downstream execution see line 86
methods_list = {0: cgr, 1: num_mapping_PP, 2: num_mapping_Int, 3: num_mapping_IntN, 4: num_mapping_Real, 5: num_mapping_Doublet, 6: num_mapping_Codons, 7: num_mapping_Atomic,
                8: num_mapping_EIIP, 9: num_mapping_AT_CG, 10: num_mapping_justA, 11: num_mapping_justC, 12: num_mapping_justG, 13: num_mapping_justT, 14: 'PuPyCGR', 15: '1DPuPyCGR'}
# change method number referring the variable above (between 0 and 15)
method_num = 6
k_val = 4  # used only for CGR-based representations(if methodNum=1,15,16)
## Not currently implemented, for future development
# test_set = None 
# seq_to_test = 0
# min_seq_len = 0
# max_seq_len = 0
# frags_per_seq = 1

method = methods_list.get(method_num)
seq_dict, number_of_clusters, total_seq, cluster_dict = preprocessing(data_set)
# variable holding all the keys (accession numbers) for corresponding clusters
keys = list(cluster_dict.keys())
values = list(cluster_dict.values())
# Could be parallelized in the future
seqs_length = [len(seq_dict[keys[i]].seq) for i in range(total_seq)]
med_len = median(seqs_length)
labels = [cluster_dict[x] for x in seq_dict.keys()]
fft_output_list = []
abs_fft_output_list = []
cgr_output_list = []

def compute_cgr_PuPyCGR(seq_index):
    # seqs is the sequence database, it is being called by accession number
    # (keys) which is being iterated over all seq_index,
    # (count of all sequences in uploaded dataset) .seq calls the string
    seq_new = str(seq_dict[keys[seq_index]].seq)
    if method_num == 14:
        seq_new = seq_new.replace('G', 'A')
        seq_new = seq_new.replace('C', 'T')
    cgr_output = cgr(seq_new, 'ACGT', k_val)  # shape:[2^k, 2^k]
    # shape:[2^k, 2^k] # may not be appropriate to take by column
    fft_output = fft.fft(cgr_output, axis=0)
    abs_fft_output = np.abs(fft_output.flatten())
    return abs_fft_output, fft_output, cgr_output  # flatted into 1d array


def compute_1DPuPyCGR(seq_index):
    seq_new = str(seq_dict[keys[seq_index]].seq)
    # creates PuPyCGR
    seq_new = seq_new.replace('G', 'A')
    seq_new = seq_new.replace('C', 'T')
    cgr_raw = cgr(seq_new, 'ACGT', k_val)
    # takes only the last (bottom) row but all columns of cgr to make 1DPuPyCGR
    cgr_output = cgr_raw[-1, :]
    # shape:[1, 2^k] # may not be appropriate to take by column
    fft_output = fft.fft(cgr_output, axis=0)
    abs_fft_output = np.abs(fft_output.flatten())
    return abs_fft_output, fft_output, cgr_output  # flatted into 1d array


def one_dimensional_num_mapping_wrapper(seq_index):
    # normalize sequences to median sequence length of cluster
    seq_new = str(seq_dict[keys[seq_index]].seq)
    if len(seq_new) >= med_len:
        seq_new = seq_new[0:round(med_len)]
    num_seq = method(seq_new)
    if len(num_seq) < med_len:
        pad_width = int(med_len - len(num_seq))
        num_seq = pywt.pad(num_seq, pad_width, 'antisymmetric')[pad_width:]
    fft_output = fft.fft(num_seq)
    abs_fft_output = np.abs(fft_output.flatten())
    return abs_fft_output, fft_output


# this can be replaced by Biopython
def compute_pearson_coeffient(x, y):
    r = pearsonr(x, y)[0]
    normalized_r = (1-r)/2
    return normalized_r


def compute_pearson_coeffient_wrapper(i, j, abs_fft_output_list):
    # print(abs_fft_output_list)
    # x = abs_fft_output_list[i]
    # y = abs_fft_output_list[indices[1]]
    # print(x)
    # print(y)
    return compute_pearson_coeffient(abs_fft_output_list[i], abs_fft_output_list[j])


def phylogenetic_tree(distance_matrix):
    names = keys
    matrix = triangle_matrix
    distance_matrix = DistanceMatrix(names, matrix)
    constructor = DistanceTreeConstructor()
    # neighbour joining tree not currently output
    # nj_tree = constructor.nj(distance_matrix)
    # neighbour_joining_tree = nj_tree.format('newick')
    upgma = constructor.upgma(distance_matrix)
    upgma_tree = upgma.format('newick')
    print(upgma_tree, file=tree_print)
    # can add code here for visualization with matplotlib


if __name__ == '__main__':
    pool = Pool()
    print('Generating numerical sequences, applying DFT, computing magnitude spectra .... \n')
    if method_num == 0 or method_num == 14:
        for abs_fft_output, fft_output, cgr_output in pool.map(compute_cgr_PuPyCGR, range(total_seq)):
            abs_fft_output_list.append(abs_fft_output)
            fft_output_list.append(fft_output)
            cgr_output_list.append(cgr_output)
    elif method_num == 15:
        for abs_fft_output, fft_output, cgr_output in pool.map(compute_1DPuPyCGR, range(total_seq)):
            abs_fft_output_list.append(abs_fft_output)
            fft_output_list.append(fft_output)
            cgr_output_list.append(cgr_output)
    else:
        for abs_fft_output, fft_output in pool.map(one_dimensional_num_mapping_wrapper, range(total_seq)):
            abs_fft_output_list.append(abs_fft_output)
            fft_output_list.append(fft_output)
       
    print('Building distance matrix')
    distance_matrix = np.zeros(shape=(total_seq, total_seq))
    distance_matrix = pool.starmap(partial(compute_pearson_coeffient_wrapper, abs_fft_output_list=abs_fft_output_list), ((
        i, j) for i in range(total_seq) for j in range(total_seq)))
    distance_matrix = np.array(distance_matrix).reshape(total_seq, total_seq)
    # change filename to unique ID
    np.savetxt('./plots/distmat.txt', distance_matrix)

    print('Building Phylogenetic Trees')
    matrix_list = distance_matrix.tolist()
    triangle_matrix = []
    # loop to create triangle matrix as nested list
    for i in range(total_seq):
        row = []
        row = (matrix_list[i][0:i+1])
        triangle_matrix.append(row)
    # change filename to unique ID
    tree_print = open('./plots/upgma.tree', 'a')
    phylogenetic_tree(triangle_matrix)
    tree_print.close()

    print('Scaling & data visualisation')
    scaled_distance_matrix = dimReduction(distance_matrix, n_dim=3, method='pca')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(scaled_distance_matrix[:, 0], scaled_distance_matrix[:, 1], scaled_distance_matrix[:, 2])
    plt.savefig('./plots/multidimensional plot.png')


    # Classification
    print('Performing classification .... \n')
    folds = 10
    if (total_seq < folds):
        folds = total_seq
    mean_accuracy, accuracy = classify_dismat(distance_matrix, labels, folds, total_seq)
    # accuracy,avg_accuracy, clNames, cMat
    # accuracies = [accuracy, avg_accuracy];
    print('Classification accuracy 5 classifiers\n', accuracy)
    print('**** Processing completed ****\n')
    pool.close()
