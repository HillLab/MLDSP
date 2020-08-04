function [pMat,mList1,mList2,mList3,mList4,mList5,mList6] = classifyTestSeqExternMisList(AcNmbTest,numMethod,disMat,alabels,SeqTest,lg,clusterNames,kVal,mLen)
    numTestSeq = length(SeqTest);
    nSeq = cell(1,numTestSeq);
    fVec = cell(1,numTestSeq);
    lgNew = cell(1,numTestSeq);
    totalSeq = length(disMat);
    parfor r=1:numTestSeq
        Sq = upper(SeqTest{r});
        
        if(numMethod==1 || numMethod>=15)             
            if(numMethod == 1)
                nSeq{r} = cgr(Sq,'ACGT',kVal);
                lgN = abs(fft(nSeq{r}));
                lgNew{r} = reshape(lgN,1,[]); 
            elseif(numMethod==15) 
                Sq = regexprep(Sq,'G','A');
                Sq = regexprep(Sq,'C','T'); 
                nSeq{r} = cgr(Sq,'ACGT',kVal);
                lgN = abs(fft(nSeq{r}));
                lgNew{r} = reshape(lgN,1,[]); 
            else
                cgrLen = power(2,kVal);
                Sq = regexprep(Sq,'G','A');
                Sq = regexprep(Sq,'C','T'); 
                nmVal = cgr(Sq,'ACGT',kVal);
                nSeq{r} = nmVal(cgrLen,1:cgrLen);  
                lgNew{r} = abs(fft(nSeq{r}));
            end              
        else    
            I = mLen-length(Sq);
            if(I<0)
                Sq=Sq(1:mLen);
            end                
            if(numMethod==2) 
                nSq = numMappingPP(Sq);
            elseif(numMethod==3)
                nSq = numMappingInt(Sq);
            elseif(numMethod==4)
                nSq = numMappingIntN(Sq);
            elseif(numMethod==5)
                nSq = numMappingReal(Sq);
            elseif(numMethod==6)
                nSq = numMappingDoublet(Sq);
            elseif(numMethod==7)
                nSq = numMappingCodons(Sq);
            elseif(numMethod==8)
                nSq = numMappingAtomic(Sq);
            elseif(numMethod==9)
                nSq = numMappingEIIP(Sq);
            elseif(numMethod==10)
                nSq = numMappingAT_CG(Sq);
            elseif(numMethod==11)
                nSq = numMappingJustA(Sq);
            elseif(numMethod==12)
                nSq = numMappingJustC(Sq);
            elseif(numMethod==13)
                nSq = numMappingJustG(Sq);
            elseif(numMethod==14) 
                nSq = numMappingJustT(Sq);          
            end
            
            if(I>0)
               nsTemp = wextend('1','asym',nSq,I);
               nSq = nsTemp((I+1):length(nsTemp)); 
            end
            
%             I = mLen-length(nSq);
%             if(I>0)
%                 nsTemp = wextend('1','asym',nSq,I);
%                 nsNew = nsTemp((I+1):length(nsTemp));
%             elseif(I<0)
%                 nsNew=nSq(1:mLen);
%             else
%                 nsNew = nSq;
%             end
%             nSeq{r} = nsNew;  
            
            nSeq{r} = nSq; 	
		    lgNew{r} = abs(fft(nSeq{r}));  
        end
%         fNew = fft(nSeq{r});    
%         lgNew = abs(fNew); 
%         testV = [];
%         for d=1:length(lg)
%             aa=(1-corrcoef(lgNew,lg{d}))/2;
%             testV = [testV aa(1,2)];
%         end
%         fVec{r} = testV;
    end
    lgg = [lg lgNew];
    fmm=cell2mat(lgg(:));
    disMattt = f_dis(fmm,'cor',0,1);
    sId = totalSeq+1;
    eId = length(disMattt);
    testVV = disMattt(sId:eId,1:totalSeq);
    
    cn = unique(alabels);
    %discriminant
    cModel1 = fitcdiscr(...
            disMat, ...
            alabels, ...
            'DiscrimType', 'linear', ...
            'Gamma', 0, ...
            'FillCoeffs', 'off', ...
            'ClassNames', cn);

    %Linear SVM
    template = templateSVM(...
                'KernelFunction', 'linear', ...
                'PolynomialOrder', [], ...
                'KernelScale', 'auto', ...
                'BoxConstraint', 1, ...
                'Standardize', true);
    cModel2 = fitcecoc(...
                disMat, ...
                alabels, ...
                'Learners', template, ...
                'Coding', 'onevsone', ...
                'ClassNames', cn);
  
    %QSVM   
    template = templateSVM(...
                'KernelFunction', 'polynomial', ...
                'PolynomialOrder', 2, ...
                'KernelScale', 'auto', ...
                'BoxConstraint', 1, ...
                'Standardize', true);
    cModel3 = fitcecoc(...
                disMat, ...
                alabels, ...
                'Learners', template, ...
                'Coding', 'onevsone', ...
                'ClassNames', cn); 

    %Fine KNN
    cModel4 = fitcknn(...
                disMat, ...
                alabels, ...
                'Distance', 'Euclidean', ...
                'Exponent', [], ...
                'NumNeighbors', 1, ...
                'DistanceWeight', 'Equal', ...
                'Standardize', true, ...
                'ClassNames', cn);

    %subspace-discriminant
    subspaceDimension = max(1, min(74, length(disMat) - 1));
    cModel5 = fitcensemble(...
                disMat, ...
                alabels, ...
                'Method', 'Subspace', ...
                'NumLearningCycles', 30, ...
                'Learners', 'discriminant', ...
                'NPredToSample', subspaceDimension, ...
                'ClassNames', cn);
       
    %subspace knn
    subspaceDimension = max(1, min(74, length(disMat) - 1));
    cModel6 = fitcensemble(...
                disMat, ...
                alabels, ...
                'Method', 'Subspace', ...
                'NumLearningCycles', 30, ...
                'Learners', 'knn', ...
                'NPredToSample', subspaceDimension, ...
                'ClassNames', cn);  
    
    mList1=cell(2,numTestSeq);
    mList2=cell(2,numTestSeq);
    mList3=cell(2,numTestSeq);
    mList4=cell(2,numTestSeq);
    mList5=cell(2,numTestSeq);
    mList6=cell(2,numTestSeq);        
    pMat = zeros(6,length(clusterNames));
    for s=1:numTestSeq
        testV = testVV(s,1:totalSeq);%testV = fVec{s};
        clabel = predict(cModel1,testV);
        pMat(1,clabel)= pMat(1,clabel)+1;
        clabe2 = predict(cModel2,testV);
        pMat(2,clabe2)= pMat(2,clabe2)+1;
        clabe3 = predict(cModel3,testV);
        pMat(3,clabe3)= pMat(3,clabe3)+1;
        clabe4 = predict(cModel4,testV);
        pMat(4,clabe4)= pMat(4,clabe4)+1;
        clabe5 = predict(cModel5,testV);
        pMat(5,clabe5)= pMat(5,clabe5)+1;
        clabe6 = predict(cModel6,testV);
        pMat(6,clabe6)= pMat(6,clabe6)+1;
        mList1{1,s}=AcNmbTest{s};
        mList2{1,s}=AcNmbTest{s};
        mList3{1,s}=AcNmbTest{s};
        mList4{1,s}=AcNmbTest{s};
        mList5{1,s}=AcNmbTest{s};
        mList6{1,s}=AcNmbTest{s};
        mList1{2,s}=clusterNames{clabel};
        mList2{2,s}=clusterNames{clabe2};
        mList3{2,s}=clusterNames{clabe3};
        mList4{2,s}=clusterNames{clabe4};
        mList5{2,s}=clusterNames{clabe5};
        mList6{2,s}=clusterNames{clabe6};
    end     
end

