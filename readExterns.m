function [AcNmb, Seq, numberOfClusters, clusterNames, pointsPerCluster,FNm] = readExterns(testingSet, maxClusSize)
    path = pwd;
    %testingSet = 'C:\Users\GURJIT\Documents\MATLAB\MLDSP\DataBase\ExtTherm';
    cd(testingSet);
    cList = dir;
    cList=cList(~ismember({cList.name},{'.','..'}));
    
    numberOfClusters = length(cList);
    clusterNames = cell(1,numberOfClusters);
    pts = cell(1,numberOfClusters);
    pointsPerCluster = cell(1,numberOfClusters);
    
    for i=1:numberOfClusters
        clusterNames{i} = cList(i).name;
        folderPath = strcat(cList(i).folder,'/',cList(i).name);
        cd(folderPath);
        allFiles = [dir('**/*.fasta');dir('**/*.fna');dir('**/*.txt');];%dir ('**/*.fasta');
        allFiles=allFiles(~startsWith({allFiles.name},{'.'}));
        pts{i}=length(allFiles); 
        cd(testingSet);
    end
    
    minClustSz = min(cell2mat(pts));
    
    if(maxClusSize==0)
        szLimit = minClustSz;
    else
        szLimit = maxClusSize;
    end
    
    Seq = {};
    AcNmb = {};
    FNm = {};
    
    for i=1:numberOfClusters
        folderPath = strcat(testingSet,'/',clusterNames{i});
        cd(folderPath);
        allFiles = [dir('**/*.fasta');dir('**/*.fna');dir('**/*.txt');];%dir ('**/*.fasta');
        allFiles=allFiles(~startsWith({allFiles.name},{'.'}));
        allInfo = struct2cell(allFiles);
        fileNames = allInfo(1,:);
        if(pts{i} <= szLimit)
            pointsPerCluster{i}=pts{i};
            seqTemp = cell(1,pts{i});
            acTemp = cell(1,pts{i});
            fnTemp = cell(1,pts{i});
            parfor j=1:pts{i}
                sbName = fileNames{j};
                [Header, Sequence] = fastaread(sbName);    
                Sequence = regexprep(Sequence,'[^A,^C, ^G, ^T]','','ignorecase');
                seqTemp{j} = Sequence;
                acTemp{j} = Header; 
                fnTemp{j} = sbName(1:length(sbName)-6);
            end            
        else
            pointsPerCluster{i}=szLimit;
            seqTemp = cell(1,szLimit);
            acTemp = cell(1,szLimit);
            fnTemp = cell(1,szLimit);
            rng(15,'twister');
            p = randperm(pts{i},szLimit);
            parfor j=1:szLimit
                sbName = fileNames{p(j)};
                [Header, Sequence] = fastaread(sbName);    
                Sequence = regexprep(Sequence,'[^A,^C, ^G, ^T]','','ignorecase');
                seqTemp{j} = Sequence;
                acTemp{j} = Header; 
                fnTemp{j} = sbName(1:length(sbName)-6); 
            end
        end  
        Seq = [Seq seqTemp];
        AcNmb = [AcNmb acTemp];
        FNm = [FNm fnTemp];
        cd(testingSet);
    end     
    cd(path);
end