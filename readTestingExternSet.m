function [AcNmb, Seq, pnts, FNm] = readTestingExternSet(dataSet,minSeqLen,maxSeqLen,maxClusSize)
% read fasta files in folder DataBase/'dataSet'
% make subFolders for each cluster and place respective fasta sequences insides
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gurjit Singh Randhawa  %
% Department of Computer Science,%
% Western University, Canada     %
% email: grandha8@uwo.ca         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    path = pwd;
    cd(dataSet);
    allFiles = [dir('**/*.fasta');dir('**/*.fna');];%dir ('**/*.fasta');
    allInfo = struct2cell(allFiles);
    fileNames = allInfo(1,:);
    pts = length(allFiles);
     
    if(maxClusSize==0 || pts<=maxClusSize)        
        SeqT = cell(1,pts);
        AcNmbT = cell(1,pts);
        FNmT = cell(1,pts);
        parfor j=1:pts
            sbName = fileNames{j};
            [Header, Sequence] = fastaread(sbName);    
            Sequence = regexprep(Sequence,'[^A,^C, ^G, ^T]','','ignorecase');
            SeqT{j} = Sequence;
            AcNmbT{j} = Header; 
            FNmT{j} = sbName(1:length(sbName)-6);
        end
    else
        rng(15,'twister');
        p = randperm(pts,maxClusSize);
        SeqT = cell(1,maxClusSize);
        AcNmbT = cell(1,maxClusSize);
        FNmT = cell(1,maxClusSize);
        parfor j=1:maxClusSize
            sbName = fileNames{p(j)};
            [Header, Sequence] = fastaread(sbName);    
            Sequence = regexprep(Sequence,'[^A,^C, ^G, ^T]','','ignorecase');
            SeqT{j} = Sequence;
            AcNmbT{j} = Header; 
            FNmT{j} = sbName(1:length(sbName)-6); 
        end
    end
    Seq={};
    AcNmb={};
    FNm={};
    id=1;
    for i=1:length(SeqT)
        lnt = length(SeqT{i});
        if(lnt>=minSeqLen)
            if(lnt<maxSeqLen || maxSeqLen==0)
                Seq{id}=SeqT{i};
                AcNmb{id}=AcNmbT{i};
                FNm{id} = FNmT{i};
                id=id+1;
            else
                sln = lnt-maxSeqLen+1;
                sRng = randi([1 sln],1,1);
                eRng = sRng+maxSeqLen-1;                    
                Seq{id}=SeqT{i}(sRng:eRng);
                AcNmb{id}=AcNmbT{i};
                FNm{id} = FNmT{i};
                id=id+1;                
            end
        end
    end   
    pnts=length(Seq);
    cd(path);
end

