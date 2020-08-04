function [tab,mList1,mList2,mList3,mList4,mList5,mList6] = testingExternMisList(dSet,numMethod,disMat,alabels,lg,clusterNames,kVal,mLen,minSeqLen,maxSeqLen,maxClusSize)
     warning('off','all');
    %read and clean testing data
    [AcNmbTest,SeqTest, pnts,Fnm] = readTestingExternSet(dSet,minSeqLen,maxSeqLen,maxClusSize);
    %totalSeq = length(lg);
    numberOfClusters = length(clusterNames);
%     for d=1:length(SeqTest)
%         if(length(SeqTest{d})>mLen)
%             SeqTest{d}=SeqTest{d}(1:mLen);
%         end
%     end
    
    %testing
    [pMat,mList1,mList2,mList3,mList4,mList5,mList6] = classifyTestSeqExternMisList(AcNmbTest,numMethod,disMat,alabels,SeqTest,lg,clusterNames,kVal,mLen);

    %if(totalSeq<=2000)
        clNames = {"LinearDiscriminant","LinearSVM","QuadraticSVM","FineKNN","SubspaceDiscriminant","SubspaceKNN"};
    %else
    %    clNames = {"LinearDiscriminant","LinearSVM","QuadraticSVM","FineKNN"};
    %end

    tabCols = numberOfClusters+1;
    sz = [length(clNames) tabCols];
    varNames = {'Model'};
    varNames = [varNames clusterNames];
    varTypes = {'string'};
    tmpTyp    = cell(1, numberOfClusters);
    tmpTyp(:) = {'double'};
    varTypes = [varTypes tmpTyp];
    tab = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
    tab(:,1) = cellstr(clNames.');
    tab(:,2:tabCols) = num2cell(pMat);
end
