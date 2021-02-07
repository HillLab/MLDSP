import math, sys
import numpy as np
from statistics import median, mean
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

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

def mds(dMatPath):
    """
    Takes input path to an nxn distance matrix. Performs Classical Multidimensional Scaling and returns an nx3 coordinate matrix, where each row 
    corresponds to one of the input sequences in a 5-dimensional euclidean space. It also produces a 3D plot, very rough testing (need to color coat via cluster labels etc.
    """

    # to integrate with your code you can just change it to take input the distance matrix itself instead of the path.
    
    dMat = np.loadtxt(dMatPath)

    eigValues, eigVectors = np.linalg.eig(dMat)
    idx = eigValues.argsort()[::-1][0:5]  
    selEigValues = eigValues[idx]
    selEigVectors = eigVectors[:,idx]

    if False in (selEigValues > 0):
        print("First 5 largest eigenvalues are not all positive. Exiting..")
        sys.exit(-1)

    selEigVectors = np.array(selEigVectors)

    diagValues = []
    for i in range(len(selEigValues)):
        diagValues.append(math.sqrt(eigValues[i]))
        
    diag = np.diag(diagValues)
    points = np.dot(selEigVectors,diag)

    minmaxScalingKameris = []
    for i in range(5):
        minmaxScalingKameris.append([ min(points[:,i]), max(points[:,i]) ])

    scaledPoints = []
    for i in range(len(dMat)):
        scaledPoints.append([0, 0, 0, 0, 0])
        for j in range(5):
            scaledPoints[i][j] = 2.0 *(points[i][j] - minmaxScalingKameris[j][0]) / ( minmaxScalingKameris[j][1] - minmaxScalingKameris[j][0]) - 1

    scaledPoints = np.array(scaledPoints) 

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = scaledPoints[:,0]
    y = scaledPoints[:,1]
    z = scaledPoints[:,2]

    ax.scatter(x, y, z, c='r', marker='o')

    fig.show()

    # return scaled data with first 5 dimensions

    return scaledPoints