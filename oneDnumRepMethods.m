function [ f, lg, nmValSH ] = oneDnumRepMethods(Seq, numMethod, mLen, totalSeq)
    nSeq=cell(1,totalSeq);
    nmValSH=cell(1,totalSeq);
    f=cell(1,totalSeq);
    lg=cell(1,totalSeq);
    
    parfor a = 1:totalSeq
        ns = upper(Seq{a});
        I = mLen-length(ns);
        if(I<0)
            Seq{a}=ns(1:mLen);
        end
    end
    
    
    if(numMethod==2)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingPP(Seq{a});
        end
    elseif(numMethod==3)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingInt(Seq{a});
        end
    elseif(numMethod==4)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingIntN(Seq{a});
        end
    elseif(numMethod==5)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingReal(Seq{a});
        end
    elseif(numMethod==6)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingDoublet(Seq{a});
        end
    elseif(numMethod==7)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingCodons(Seq{a});
        end
    elseif(numMethod==8)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingAtomic(Seq{a});
        end
    elseif(numMethod==9)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingEIIP(Seq{a});
        end
    elseif(numMethod==10)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingAT_CG(Seq{a});
        end
    elseif(numMethod==11)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingJustA(Seq{a});
        end
    elseif(numMethod==12)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingJustC(Seq{a});
        end
    elseif(numMethod==13)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingJustG(Seq{a});
        end
    elseif(numMethod==14)
        parfor a = 1:totalSeq
            nSeq{a} = numMappingJustT(Seq{a});
        end
    else
        fprintf('**** Please select valid numerical representation ****\n');
        exit;
    end
    
    parfor a = 1:totalSeq
        ns = nSeq{a};
        I = mLen-length(ns);
        if(I>0)
            nsTemp = wextend('1','asym',ns,I);
            nsNew = nsTemp((I+1):length(nsTemp));
        else
            nsNew = ns;
        end
        nmValSH{a} = nsNew;
        f{a} = fft(nsNew);
        lg{a} = abs(f{a});
    end       
end