import numpy as np
from numpy import matrix.mean as mean

def interClusdist(Cluster,uniqueClusters,disMat, numberOfClusters):
    avgDisB = np.zeros((numberOfClusters,numberOfClusters))
    for h in range(numberOfClusters):
        cInd[h] = (Cluster == uniqueClusters[h])
    
    for i in range(numberOfClusters)
        for j in range(i+1, numberOfClusters)
            if(i==j)            
                continue;            
            else
                dT = disMat[cInd[i],cInd[j]]
                avgDisB[i,j] = mean[m1, 1]  
                avgDisB[j,i] = avgDisB[i,j]