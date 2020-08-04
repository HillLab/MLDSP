function [ avgDisB ] = interClusDist( Cluster, uniqueClusters,disMat, numberOfClusters)    
    avgDisB = zeros(numberOfClusters,numberOfClusters);
    for h=1:numberOfClusters      
        cInd{h} = Cluster == uniqueClusters(h);
    end
    
    for i=1:numberOfClusters 
        for j=(i+1):numberOfClusters 
            if(i==j)            
                continue;            
            else
                dT = disMat(cInd{i},cInd{j});
                m1 = mean(dT.');
                avgDisB(i,j) = mean(m1);  
                avgDisB(j,i) = avgDisB(i,j);
            end
        end 
    end      
end

