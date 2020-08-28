from statistics import median, mean
import numpy as np

def length_calc(seq_list):
    """calculates length stats

    Keyword arguments:
    seq_list: a list of squence
    """
    len_list = map(len, seq_list)
    max_len = max(len_list)
    min_len = min(len_list)
    mean_len = mean(len_list)
    med_len = median(len_list)

    return max_len, min_len, mean_len, med_len

def inter_cluster_dist(clsuter,unique_clusters,distance_matrix, cluster_num):
    avg_dist = np.zeros((cluster_num,cluster_num))
    c_ind = np.zeros(cluster_num)
    for h in range(cluster_num):
        c_ind[h] = (clsuter == unique_clusters[h])
    
    for i in range(cluster_num):
        for j in range(i+1, cluster_num):
            if i==j:         
                continue           
            else:
                dT = distance_matrix[c_ind[i],c_ind[j]]
                avg_dist[i,j] = np.mean(np.transpose(dT), 1)  
                avg_dist[j,i] = avg_dist[i,j]
    return avg_dist