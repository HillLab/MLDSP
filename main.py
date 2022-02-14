# from pstats import SortKey
# import cProfile, pstats
# profiler = cProfile.Profile()
# profiler.enable()
# import tracemalloc
# tracemalloc.start(25)
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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import preprocessing as pe
import h5py
import os
import shutil
import pathlib
import io 
# mldsp imports
from classification import classify_dismat
from preprocessing import *
from helpers import *
from cgr import *
from visualisation import dimReduction


def compute_cgr(seq_index, seq_dict, keys, method_num, k_val,Result_path):
    # seqs is the sequence database, it is being called by accession number
    # (keys) which is being iterated over all seq_index,
    # (count of all sequences in uploaded dataset) .seq calls the string
    seq_new = str(seq_dict[keys[seq_index]].seq)
    # Replace complementary Purine/Pyrimidine
    if method_num == 15 or method_num == 16:
        seq_new = seq_new.replace('G', 'A')
        seq_new = seq_new.replace('C', 'T')
    cgr_raw = cgr(seq_new, 'ACGT', k_val)  # shape:[2^k, 2^k]
    # takes only the last (bottom) row but all columns of cgr to make 1DPuPyCGR
    if method_num == 16:
        cgr_output = cgr_raw[-1, :]
    else:
        cgr_output = cgr_raw
    # shape:[2^k, 2^k] # may not be appropriate to take by column
    np.save(os.path.join(Result_path,'Num_rep','cgr'+'k='+str(k_val)+'_'+str(seq_index)),cgr_output)
    fft_output = fft.fft(cgr_output, axis=0)
    abs_fft_output = np.abs(fft_output.flatten())
    return abs_fft_output, fft_output, cgr_output  # flatted into 1d array


def one_dimensional_num_mapping_wrapper(seq_index, method, seq_dict, keys, med_len):
    # normalize sequences to median sequence length of cluster
    seq_new = str(seq_dict[keys[seq_index]].seq)
    if len(seq_new) >= med_len:
        seq_new = seq_new[0:round(med_len)]
    num_seq = method(seq_new) 
    if len(num_seq) < med_len:
        pad_width = int(med_len - len(num_seq))
        num_seq = pywt.pad(num_seq, pad_width, 'antisymmetric')[pad_width:]
    np.save(os.path.join(Result_path,'Num_rep',method+str(seq_index)),num_seq)
    fft_output = fft.fft(num_seq)
    abs_fft_output = np.abs(fft_output.flatten())
    return abs_fft_output, fft_output

# def compute_pearson_coeffient(x, y, i, j):
#     r = pearsonr(x, y)[0]
#     normalized_r = (1-r)/2
#     return normalized_r

# def compute_pearson_coeffient_wrapper(abs_fft_output_list):
#     distance_matrix = (1-np.corrcoef(abs_fft_output_list))/2
#     return(distance_matrix)

# def compute_pearson_coeffient_wrapper(i, j, abs_fft_output_list):
#     # print(abs_fft_output_list)
#     # x = abs_fft_output_list[i]
#     # y = abs_fft_output_list[indices[1]]
#     # print(x)
#     # print(y)
#     return compute_pearson_coeffient(abs_fft_output_list[i], abs_fft_output_list[j],i,j)


def phylogenetic_tree(triangle_matrix):
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
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    ## User set up, should be changed to argparse
    data_set = '/Users/dolteanu/local_documents/Coding/MLDSP_dev_git/data/Primates/fastas'
    metadata = '/Users/dolteanu/local_documents/Coding/MLDSP_dev_git/data/Primates/metadata.csv'
    Run_name = 'Bacteria'
    # data_set = '/Volumes/NVME-ssd/Gisaid data/Gisaid data 01:11:22/hcov_global_2022-01-09_23-30/Testing/Fastas'
    # metadata = '/Volumes/NVME-ssd/Gisaid data/Gisaid data 01:11:22/hcov_global_2022-01-09_23-30/Testing/gisaid_metadata<20.csv'
    # Run_name = "Nextstrain_gisaid<20_cgrtest"
    Result_path = pathlib.Path(os.path.join('./Results',Run_name))
    if os.path.exists(Result_path):
        shutil.rmtree(Result_path)
        os.makedirs(os.path.join(Result_path,"Num_rep",'abs_fft'))
    else:
        os.makedirs(os.path.join(Result_path,"Num_rep",'abs_fft'))
    # Dictionary order & names dependent for downstream execution 
    methods_list = {1: num_mapping_PP, 2: num_mapping_Int, 3: num_mapping_IntN, 4: num_mapping_Real, 5: num_mapping_Doublet, 6: num_mapping_Codons, 7: num_mapping_Atomic,
                    8: num_mapping_EIIP, 9: num_mapping_AT_CG, 10: num_mapping_justA, 11: num_mapping_justC, 12: num_mapping_justG, 13: num_mapping_justT, 14: 'cgr',15: 'PuPyCGR', 16: '1DPuPyCGR'}
    # Change method number referring the variable above (between 1 and 16)
    method_num = 14
    k_val = 5  # used only for CGR-based representations(if methodNum=14,15,16)
    ## End of user set up


    ## Not currently implemented, for future development
    # test_set = None 
    # seq_to_test = 0
    # min_seq_len = 0
    max_clust_size = 10000000
    # frags_per_seq = 1
    method = methods_list.get(method_num)
    seq_dict, total_seq, cluster_dict, cluster_stats = preprocessing(data_set,max_clust_size, metadata)
    # print(cluster_stats)
    
    # Could be parallelized in the future

    # variable holding all the keys (accession numbers) for corresponding clusters
    keys = list(seq_dict.keys())
    # values = list(cluster_dict.values())
    #seq dict keys have to be in same order as cluster dict keys (see above)
    labels = [cluster_dict[x] for x in seq_dict.keys()]
    np.save(os.path.join(Result_path,'labels'),np.array(labels))
    seqs_length = [len(seq_dict[keys[i]].seq) for i in range(total_seq)]
    med_len = median(seqs_length)
    print('Mean seq length: '+ str(med_len))
    fft_output_list = []
    abs_fft_output_list = []
    cgr_output_list = []
    seq_new_list = []
    seq_list = []
    print('Run_name', Run_name,'\nMethod:',method,k_val,'\nMean seq length: '+ str(med_len),'\nCluster sizes:'+str(cluster_stats),file=open(os.path.join(Result_path,'Run_data.txt'),'x'))
    
    print('Generating numerical sequences, applying DFT, computing magnitude spectra .... \n')
    # for seq_index in range(total_seq):
    #     if method_num == 14 or method_num == 15 or method_num == 16:
    #         abs_fft_output, fft_output, cgr_output = compute_cgr(seq_index,seq_dict,keys,method_num,k_val)
    #         abs_fft_output_list.append(abs_fft_output)
    #         fft_output_list.append(fft_output)
    #         cgr_output_list.append(cgr_output)
    #         # seq_new_list.append(seq_new)
    #         # seq_list.append(seq)
    #     else:
    #         abs_fft_output, fft_output, seq_new, seq = one_dimensional_num_mapping_wrapper(seq_index)
    #         fft_output_list.append(fft_output)
    #         abs_fft_output_list.append(abs_fft_output)
    #         seq_new_list.append(seq_new)
    #         seq_list.append(seq)
            
    pool = Pool()
    if method_num == 14 or method_num == 15 or method_num == 16:
        for abs_fft_output, fft_output, cgr_output in pool.map(partial(compute_cgr, seq_dict=seq_dict, keys=keys, method_num=method_num, k_val=k_val,Result_path=Result_path), range(total_seq)):
            abs_fft_output_list.append(abs_fft_output)
            fft_output_list.append(fft_output)
            cgr_output_list.append(cgr_output)
    else:
        for abs_fft_output, fft_output in pool.map(partial(one_dimensional_num_mapping_wrapper, method = method, seq_dict=seq_dict, keys=keys, med_len=med_len,Result_path=Result_path), range(total_seq)):
            abs_fft_output_list.append(abs_fft_output)
            fft_output_list.append(fft_output)
    pool.close()

    if method_num == 14 or method_num == 15:
        plt.matshow(cgr_output_list[0],cmap=cm.gray_r)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(Result_path,'cgr_0.png'))

    print('Building distance matrix')
    
    # # hdf5 implementation
    # with h5py.File("distmat"+Run_name+".hdf5", "w") as dist_mat_file:
    #     distance_matrix = dist_mat_file.create_dataset("dist_mat", (total_seq, total_seq))
    #     for i in range(total_seq):
    #         for j in range(total_seq):
    #             distance_matrix[i,j] = compute_pearson_coeffient_wrapper(i,j,abs_fft_output_list)
    # reading file if already made
    # # rand = h5py.File('/Users/dolteanu/local_documents/Coding/MLDSP_dev_git/distmat.hdf5', 'r')
    # # distance_matrix = rand['dist_mat']

    # #parallel implementation
    # for normalized_r, i, j in pool.starmap(partial(compute_pearson_coeffient_wrapper, abs_fft_output_list=abs_fft_output_list), ((i, j) for i in range(total_seq) for j in range(total_seq)),chunksize=total_seq):
    #     distance_matrix[i,j] = normalized_r
    # distance_matrix = compute_pearson_coeffient_wrapper(abs_fft_output_list)
    # # Numpy implementation
    distance_matrix = (1-np.corrcoef(abs_fft_output_list))/2
    np.save(os.path.join(Result_path,'dist_mat'), distance_matrix)
    
    print('Building Phylogenetic Trees')
    
    # matrix_list = distance_matrix.tolist()
    # triangle_matrix = []
    # # loop to create triangle matrix as nested list
    # for i in range(total_seq):
    #     row = []
    #     row = (matrix_list[i][0:i+1])
    #     triangle_matrix.append(row)
    # # change filename to unique ID
    # tree_print = open(os.path.join(Result_path,'upgma.tree'), 'a')
    # phylogenetic_tree(triangle_matrix)
    # tree_print.close()

    # print('Visualizing fourier spectra')
    
    # # Run this cell for 1D numerical representations Fourier spectrum
    # median = int(med_len)
    # # fftshift the magnitude spectrum (required to visualize)
    # shifted_abs_fft = fft.fftshift(abs_fft_output_list[0])
    # fig3, ax = plt.subplots(nrows=1, ncols=1)
    # # normalize the x axis to the length with -ve on left & +ve on right 
    # fVals = np.arange(start=-median/2, stop=median/2)/median
    # ax.plot(fVals,shifted_abs_fft)
    # ax.set_title('Double Sided FFT - with FFTShift')
    # ax.set_xlabel('Normalized Frequency')
    # ax.set_ylabel('|DFT Values|')
    # ax.autoscale(enable=True, axis='x', tight=True)
    # ax.set_xticks(np.arange(-0.5, 0.5+0.1,0.1))
    
    # #Run this for 2D CGR fourier spectrum
    # length = (2**k_val)**2 
    # shifted_abs_fft = fftshift(abs_fft_output_list[0])
    # fig4, ax = plt.subplots(nrows=1, ncols=1)
    # fVals = np.arange(start=-length/2, stop=length/2)/length
    # ax.plot(fVals,shifted_abs_fft)
    # ax.set_title('Double Sided FFT - with FFTShift')
    # ax.set_xlabel('Length Normalized Frequency')
    # ax.set_ylabel('Magnitude')
    # ax.autoscale(enable=True, axis='x', tight=True)
    # ax.set_xticks(np.arange(-0.5, 0.5+0.1,0.1))
    # #Run only for CGR plotting
    # plt.matshow(cgr_output_list[0],cmap=cm.gray_r)
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('./plots/cgr.png')


    print('Scaling & data visualisation')
    le = pe.LabelEncoder()
    le.fit(labels)
    labs = le.transform(labels)
    scaled_distance_matrix = dimReduction(distance_matrix, n_dim=3, method='pca')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(scaled_distance_matrix[:, 0], scaled_distance_matrix[:, 1], scaled_distance_matrix[:, 2],c=labs)
    plt.savefig(os.path.join(Result_path,'MoDmap.png'))


    # Classification
    print('Performing classification .... \n')
    folds = 10
    if (total_seq < folds):
        folds = total_seq
    mean_accuracy, accuracy, cmatrix, misClassifiedDict = classify_dismat(distance_matrix, labels, folds, total_seq)
    # accuracy,avg_accuracy, clNames, cMat
    # accuracies = [accuracy, avg_accuracy];
    print('\n10X cross validation classifier accuracies',accuracy,file=open(os.path.join(Result_path,'Run_data.txt'),'a'))
    print('**** Processing completed ****\n')
    # profiler.disable()
    # with open('profile2.txt','w') as p:
    #     stats = pstats.Stats(profiler,stream=p)
    #     stats.strip_dirs()
    #     stats.sort_stats(SortKey.CUMULATIVE, SortKey.TIME)
    #     stats.dump_stats('./profile2.prof')
    #     stats.print_stats(30)
    #     stats.print_callers()
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno',cumulative=True)
    # snapshot.dump('mem_snapshot_parallel_dengue')
    # for stat in top_stats[:10]:
    #     print(stat,file=open('mem_profile_parallel_dengue.txt','a'))
