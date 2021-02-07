%clear workspace, config
close all;
clear all;
clc ;

dataSet = 'g__Prevotella_ce/o__Bacteroidales_exclude_g__Prevotella/'
splitDataSet = split(dataSet, '/');
parentSet = splitDataSet{1,:};
childSet = splitDataSet{2,:};
testSet = 'MAG/g__Prevotella/'
% testSet = ''

basePath = '/home/w328li/MLDSP-desktop/samples/';
if isunix & ismac
    basePath = '/Users/wanxinli/Desktop/project/MLDSP-desktop/samples/';
end
methodNum = 1
dataSetPath = strcat(basePath, dataSet)
testSetPath = strcat(basePath, testSet)
kVal = 7; % used only for CGR-based representations(if methodNum=1,15,16,17)
selectedFolder = dataSetPath;

% read sequences
fprintf('Reading sequences .... \n');
maxClusSize = 50000; %  0 if limit all cluster sizes by the size of smallest cluster;
[acN, Seq, numberOfClusters, clusterNames, pointsPerCluster,Fnm] = readExternsBac(dataSetPath,maxClusSize);
AcNmb = Fnm;
totalSeq = length(Seq);
[maxLen, minLen, meanLen, medLen] = lengthCalc(Seq);

% construct single-stranded cgr, uncomment lines 42,44,48 for double stranded stacked cgr
allCGR = cell(1,totalSeq);
for i=1:totalSeq
    ss = Seq{i};
    tCGR=zeros(2^kVal);
    for j=1:length(ss)
        sq = ss{j};
        sqSeg = regexp(sq, '[^ATCG]', 'split');
        for m=1:length(sqSeg)
            seg = sqSeg{m};
            tCGRNw=cgr(seg,'ACGT',kVal);
	    %fprintf('tCGRNw done\n');
            segComp = seqrcomplement(seg);
	    %fprintf('segComp done\n');
            tCGRNwComp = cgr(segComp,'ACGT',kVal);    
	    %fprintf('tCGRNwComp done\n');
            tCGR = tCGR+tCGRNw+tCGRNwComp;
	    %tCGR = tCGR+tCGRNw;
        end
    end  
    allCGR{i}=tCGR;
end

f = cell(1, totalSeq);
lg = cell(1, totalSeq);
for i=1:totalSeq
    f{i} = fft(allCGR{i});
    lg{i} = abs(f{i});
end

% compute distance matrix
fprintf('Computing Distance matrix .... \n');
lgl = cell(1,totalSeq);
parfor i=1:totalSeq
    lgl{i} =  reshape(lg{i},1,[]);
end
lg = lgl;
fm=cell2mat(lgl(:));
disMat = f_dis(fm,'cor',0,1);
[Y,eigvals] = cmdscale(disMat,3);

% disp(disMat);

clusterStart = pointsPerCluster;
for i=1:length(clusterStart)
    if i==1
        clusterStart{i} = 1;
    else
        clusterStart{i} = clusterStart{i-1}+pointsPerCluster{i-1};
    end
end

for i=1:length(clusterStart)
    for j=i:length(clusterStart)
        pdisMat = disMat(clusterStart{i}:(clusterStart{i}+pointsPerCluster{i}-1), clusterStart{j}:(clusterStart{j}+pointsPerCluster{j}-1));
        pdisMat = pdisMat';
        pdisMat = pdisMat(:)';
        fprintf("%s and %s avg dissimilarity is: %f\n", clusterNames{i}, clusterNames{j}, nanmean(pdisMat))
    end
end

%3D  plot
fprintf('Generating 3D plot .... \n');
index=1;
counter=1;
Cluster = zeros(1,totalSeq);
for i=1:totalSeq   
    Cluster(i)=index;
    if(counter==pointsPerCluster{index})
        index=index+1;
        counter=0;
    end
    counter= counter+1;
end
uniqueClusters  = unique(Cluster);
cmap = distinguishable_colors(numberOfClusters);
hf = figure;
hold on;
for h=1:numberOfClusters
    cIndex = Cluster == uniqueClusters(h);
    plot3(Y(cIndex,1),Y(cIndex,2),Y(cIndex,3),'.','markersize', 15, 'Color',cmap(h,:),'DisplayName',clusterNames{h});
end
% view(3), axis vis3d, box on, datacursormode on
xlabel('x'), ylabel('y'), zlabel('z')
tname = strcat(selectedFolder,' (',int2str(totalSeq),' Sequences',')');
title(tname)
hdt = datacursormode(hf);
set(hdt,'UpdateFcn',{@myupdatefcn,Y,Fnm})
legend('show');
mkdir(strcat('outputs/3D/', parentSet));
saveas(hf, strcat('outputs/3D/', parentSet, '/', childSet, '.fig'));

% create labels
clear a;
clear s;
a=[];
for i=1:numberOfClusters
    for j=1:pointsPerCluster{i}
        a=[a; i];
    end
end

% perform classification
fprintf('Performing classification .... \n');
alabels = a;
folds=10;
if (totalSeq<folds)
    folds = totalSeq;
end
if (strcmp(testSet, ''))
    [accuracy, avg_accuracy, clNames, cMat] = classificationCode(disMat,alabels, folds, totalSeq, AcNmb);
    acc = [accuracy avg_accuracy];
    s.ClassifierModel=cellstr(clNames.');
    s.Accuracy=cell2mat(acc).';
    ClassificationAccuracyScores = struct2table(s);
    disp(ClassificationAccuracyScores);
end

if (~strcmp(testSet,''))
    minSeqLen = 0
    maxSeqLen = 0
    seqToTest = 0
    [tab,mList1,mList2,mList3,mList4,mList5,mList6]=testingExternMisList(testSetPath,methodNum,disMat,alabels,lg,clusterNames,kVal,medLen,minSeqLen,maxSeqLen,seqToTest, clusterStart, dataSet);
    tabc=table2cell(tab);
    tabc=[tab.Properties.VariableNames;tabc];
    tabc = string(tabc);
    % writematrix(tabc,'testingMatrix4.xlsx');
    % writematrix(string(mList1(1)),'testingMatrix4.xlsx','Sheet',2);  
    % writematrix(string(mList2(1)),'testingMatrix4.xlsx','Sheet',3);  
    % writematrix(string(mList3(1)),'testingMatrix4.xlsx','Sheet',4);  
    % writematrix(string(mList4(1)),'testingMatrix4.xlsx','Sheet',5);  
    % writematrix(string(mList5(1)),'testingMatrix4.xlsx','Sheet',6); 
    % writematrix(string(mList6(1)),'testingMatrix4.xlsx','Sheet',7);  
end


fprintf('Process completed \n')
