% ML-DSP
% the main program that user needs to run for the results.
% default dataSet is Primates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gurjit Singh Randhawa  %
% Department of Computer Science,%
% Western University, Canada     %
% email: grandha8@uwo.ca         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear workspace
close all;
clear all;
clc ;
%warning('off','all')
%read fasta files; give datasetName with complete path
dataSet = '/Users/dolteanu/local_documents/Coding/MLDSP_github/DataBase/Primates';
testingSet = 'NoData';% change to 'NoData' if there is no testing set
%otherwise change as shown below and uncomment the testing code towards end
%testingSet = 'F:\Exterm17Dec\Test4\Test\HalophileBacteria';
seqToTest=0; % 0 to test all sequences, or any value upto number of available sequences
minSeqLen = 0;    %shorter sequences will be omited, change to 0 for considering shorter sequences
maxSeqLen = 0; %random fragment from longer sequences will be used, change to 0 for using original length of all sequences
fragsPerSeq = 1; %shorter sequences will be considered as they are, multiple non-overlapping fragments will be used for longer sequences
%default is 1 fragment of length = maxSeqLen per sequence; if fragsPerSeq>1
%then for each sequence, script selects non-overlapping fragments upto 'selected value' minimum(fragsPerSeq, maximum possible fragments)

%select method
methodsList = {'CGR(ChaosGameRepresentation)','Purine-Pyrimidine','Integer','Integer (other variant)','Real','Doublet','Codons','Atomic','EIIP','PairedNumeric','JustA','JustC','JustG','JustT','PuPyCGR','1DPuPyCGR'};
methodNum=14; %change method number referring the variable above (between 1 and 16)
kVal = 4; % used only for CGR-based representations(if methodNum=1,15,16)

selectedFolder = dataSet;
fprintf('Reading sequences .... \n');%load('Bac2500seq.mat');

%**********cluster size options*****
%run the following with a limit on the cluster size (randomly select the sequences)
%maxClusSize = 0 limit all cluster sizes by the size of smallest cluster;
%(removing smaller sequences may change the cluster size later)
%otherwise reduce the size of larger clusters by the assigned value
maxClusSize = 5000;
[AcNmb, Seq, numberOfClusters, clusterNames, pointsPerCluster,Fnm] = readExterns(dataSet,maxClusSize);

%***Misclassified labels required***
% value 0 runs in parallel and hence faster
% 0 - for confusion matrix and classification accuracy scores
% 1 - for misclassfied labels (in addition to above)
msLblReq = 0;

%***********************************

%preProcess data
fprintf('Preprocessing data .... \n');
[Seq, AcNmb, pointsPerCluster, totalSeq] = preprocessData(Seq,AcNmb,numberOfClusters,pointsPerCluster,minSeqLen,maxSeqLen,fragsPerSeq);

%calculate length stats
[maxLen, minLen, meanLen, medLen] = lengthCalc(Seq);

%numerical sequences, length normalized by median length
%code can be modified to use other length stats for length normalization
mLen = medLen;
nmValSH=cell(1,totalSeq);
f=cell(1,totalSeq);
lg=cell(1,totalSeq);
sequence=cell(1,totalSeq);
fprintf('Generating numerical sequences, applying DFT, computing magnitude spectra .... \n');

if(methodNum==1 || methodNum==15)
    parfor a = 1:totalSeq
        sq = upper(Seq{a});
        if(methodNum==15)
            sq = regexprep(sq,'G','A');
            sq = regexprep(sq,'C','T');
        end
        nsNew = cgr(sq,'ACGT',kVal);
        nmValSH{a} =  nsNew;
        f{a} = fft(nsNew);
        lg{a} = abs(f{a});
        sequence{a}=sq
    end

    %distance calculation by Pearson correlation coefficient
    fprintf('Computing Distance matrix .... \n');
    lgl = cell(1,totalSeq);
    parfor i=1:totalSeq
        lgl{i} =  reshape(lg{i},1,[]);
    end
    lg = lgl;
    fm=cell2mat(lgl(:));
    disMat = f_dis(fm,'cor',0,1);
else
    if(methodNum==16)
        cgrLen = power(2,kVal);
        f=cell(1,totalSeq);
        lg=cell(1,totalSeq);
        parfor a = 1:totalSeq
            sq = upper(Seq{a});
            sq = regexprep(sq,'G','A');
            sq = regexprep(sq,'C','T');
            sequence{a}=sq
            nmVal = cgr(sq,'ACGT',kVal);
            nmValSH{a} = nmVal(cgrLen,1:cgrLen);
            f{a} = fft(nmValSH{a});
            lg{a} = abs(f{a});
        end
    else
        [ f, lg, nmValSH ] = oneDnumRepMethods(Seq, methodNum, mLen, totalSeq);
    end
    fprintf('Computing Distance matrix .... \n');
    fm=cell2mat(lg(:));
    disMat = f_dis(fm,'cor',0,1);
end

%Multi-dimensional Scaling
fprintf('Performing Multi-dimensional scaling .... \n');
[Y,eigvals] = cmdscale(disMat,3);

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
view(3), axis vis3d, box on, datacursormode on
xlabel('x'), ylabel('y'), zlabel('z')
tname = strcat(selectedFolder,' (',int2str(totalSeq),' Sequences',')');
title(tname)
hdt = datacursormode(hf);
set(hdt,'UpdateFcn',{@myupdatefcn,Y,Fnm})
legend('show');


%Phylogenetic Tree
treeLbl=cell(1,length(AcNmb));
id=1;
for i=1:numberOfClusters
    for j=1:pointsPerCluster{i}
        aid=strcat(clusterNames{i},'_',AcNmb{id});
        %aid=AcNmb{id};
        treeLbl{id} = regexprep(aid,'[^a-zA-Z0-9]','_');
        id=id+1;
    end
end
UPGMAtree = seqlinkage(disMat,'UPGMA',treeLbl);
phytreewrite('UPTree.tree',UPGMAtree,'Branchnames',false);

NJtree = seqneighjoin(disMat,'equivar',treeLbl);
phytreewrite('NJTree.tree',NJtree,'Branchnames',false);


%Classification Code
clear a;
clear s;

a=[];
for i=1:numberOfClusters
    for j=1:pointsPerCluster{i}
        a=[a; i];
    end
end
ATestlg = [disMat a];
rng(15,'twister');

alabels = a;
fprintf('Performing classification .... \n');
folds=10;
if (totalSeq<folds)
    folds = totalSeq;
end
tic
if(msLblReq==0)
    [accuracy, avg_accuracy, clNames, cMat] = classificationCode(disMat,alabels, folds, totalSeq);
else
    [accuracy, avg_accuracy, clNames, cMat, mls] = classificationCodeMisList(disMat,alabels, folds, totalSeq, AcNmb, clusterNames);
end
toc
acc = [accuracy avg_accuracy];
s.ClassifierModel=cellstr(clNames.');
s.Accuracy=cell2mat(acc).';
ClassificationAccuracyScores = struct2table(s)

fprintf('**** Processing completed ****\n');

%uncomment the respective parts if you want to write results to disk
% OR you want to test additional dataset

% cpth = pwd;
% i=1;
% while(true)
%     nPath = strcat(cpth,'\TestRun',num2str(i));
%     if ~exist(nPath, 'dir')
%         mkdir(nPath)
%         cd(nPath);
%         break;
%     end
%     i=i+1;
% end
%
% % write classification results
% cc = table2cell(ClassificationAccuracyScores);
% cscore = string(cc);
% cl= string(clusterNames);
% LbRows = {'True_Predictor',cl{1:end}};
% writematrix(cscore,'classScore3.xlsx');
%
% for i=1:length(cMat)
%     cm = cMat{i};
%     cMatrix = 0;
%     for k = 1:folds
%         cMatrix = cMatrix+cm{k};
%     end
%     cMatrix = [cl.' cMatrix];
%     cMatrix = [LbRows;cMatrix];
%     sheetNum = i+1;
%     writematrix(cMatrix,'classScore3.xlsx','Sheet',sheetNum);
% end
%
% %write distance matrix
% Lbs=[];
% for i=1:numberOfClusters
%     for j=1:pointsPerCluster{i}
%         Lbs=[Lbs; string(clusterNames{i})];
%     end
% end
% acc = AcNmb.';
% disAll = [Lbs acc disMat];
% LbRow = Lbs.';
% LbRow = {'ClusterName',' ',LbRow{1:end}};
% acRow = {' ','AccessionNumber',AcNmb{1:end}};
% disAll = [LbRow;acRow;disAll];
% writematrix(disAll,'distMatr2.xlsx');
%
% %inter-cluster distance
% avgDisB  = interClusDist( Cluster, uniqueClusters,disMat, numberOfClusters);
% inClusDis = [cl.' avgDisB];
% inClusDis = [LbRows;inClusDis];
% writematrix(inClusDis,'IntClustDist2.xlsx');
%
% %write MoDMap
% print(hf,'MoDMap','-dpng');
% %
% % %testing part
% if(~strcmp(testingSet,'NoData'))
%     [tab,mList1,mList2,mList3,mList4,mList5,mList6]=testingExternMisList(testingSet,methodNum,disMat,alabels,lg,clusterNames,kVal,mLen,minSeqLen,maxSeqLen,seqToTest);
%     tabc=table2cell(tab);
%     tabc=[tab.Properties.VariableNames;tabc];
%     tabc = string(tabc);
%     writematrix(tabc,'testingMatrix4.xlsx');
%     writematrix(string(mList1.'),'testingMatrix4.xlsx','Sheet',2);
%     writematrix(string(mList2.'),'testingMatrix4.xlsx','Sheet',3);
%     writematrix(string(mList3.'),'testingMatrix4.xlsx','Sheet',4);
%     writematrix(string(mList4.'),'testingMatrix4.xlsx','Sheet',5);
%     writematrix(string(mList5.'),'testingMatrix4.xlsx','Sheet',6);
%     writematrix(string(mList6.'),'testingMatrix4.xlsx','Sheet',7);
% end
%
% cd(cpth);
