function [SeqNw,AcNw,ptsNw,tSeq] = preprocessData(Seq,AcNmb,numberOfClusters,pointsPerCluster,minSeqLen,maxSeqLen,fragsPerSeq)
    SeqNw = {};
    AcNw = {};
    ptsNw = cell(1,numberOfClusters);
    
    sId = 1;
    eId = pointsPerCluster{1};
    id=1;
    
    for i=1:numberOfClusters
        cnt=0;        
        for j=sId:eId            
            if(length(Seq{j})<minSeqLen)
                continue;
            end
            if(maxSeqLen==0 || length(Seq{j})<=maxSeqLen)
                SeqNw{id}=Seq{j};
                AcNw{id}=AcNmb{j};
                id=id+1;
                cnt=cnt+1;
            else
                seqln = length(Seq{j});                
                if(fragsPerSeq==1)
                    sln = seqln-maxSeqLen+1;
                    sRng = randi([1 sln],1,1);
                    eRng = sRng+maxSeqLen-1;                    
                    SeqNw{id}=Seq{j}(sRng:eRng);
                    AcNw{id}=AcNmb{j};
                    id=id+1;
                    cnt=cnt+1;                    
                else
                    mx = floor(seqln/maxSeqLen);
                    numFrag = min(mx,fragsPerSeq);
                    sRng=1;
                    for m=1:numFrag
                        eRng = sRng+maxSeqLen-1; 
                        SeqNw{id}=Seq{j}(sRng:eRng);
                        AcNw{id}=strcat(AcNmb{j},'Frag',num2str(m));
                        sRng = eRng+1;                        
                        id=id+1;
                        cnt=cnt+1; 
                    end                    
                end
            end
        end
        sId = eId+1;
        if(i<numberOfClusters)
            eId = (sId+pointsPerCluster{i+1})-1;
        end
        ptsNw{i}=cnt;
        tSeq=length(SeqNw);
    end
end

