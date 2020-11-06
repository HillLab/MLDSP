import numpy as np

def numMappingAT_CG(sq):
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
        t = sq[k]
        if t == 'A':
            numSeq[k] = 1
        elif t == 'C':
            numSeq[k] = -1
        elif t == 'G':
            numSeq[k] = -1
        elif t == 'T':
            numSeq[k] = 1
        else:
            pass
    return numSeq
        
def numMappingJustA(sq):
     a = "A"
     length = len(sq)
     numSeq = np.zeros(length)
     for k in range(0, length):
         t = sq[k]
         if t.upper() == a:
             numSeq[k] = 1
         else:
             pass
     return numSeq
         
def numMappingJustC(sq):
     c = "C"
     length = len(sq)
     numSeq = np.zeros(length)
     for k in range(0, length):
         t = sq[k]
         if t.upper() == c:
             numSeq[k] = 1
         else:
             pass
     return numSeq
         
def numMappingJustG(sq):
     g = "G"
     length = len(sq)
     numSeq = np.zeros(length)
     for k in range(0, length):
         t = sq[k]
         if t.upper() == g:
             numSeq[k] = 1
         else:
             pass
     return numSeq

def numMappingJustT(sq):
     t_ = "T"
     length = len(sq)
     numSeq = np.zeros(length)
     for k in range(0, length):
         t = sq[k]
         if t.upper() == t_:
             numSeq[k] = 1
         else:
             pass
     return numSeq

           
def numMappingReal(sq):
     length = len(sq)
     numSeq = np.zeros(length)
     for k in range(0, length):
         t = sq[k]
         if t.upper() == "A":
             numSeq[k] = -1.5
         elif t.upper() == "C":
             numSeq[k] = 0.5
         elif t.upper() == "G":
             numSeq[k] = -0.5
         elif t.upper() == "T":
             numSeq[k] = 1.5           
         else:
             pass
     return numSeq


def numMappingPP(sq):
     length = len(sq)
     numSeq = np.zeros(length)
     for k in range(0, length):
         t = sq[k]
         if t.upper() == "A":
             numSeq[k] = -1
         elif t.upper() == "C":
             numSeq[k] = 1
         elif t.upper() == "G":
             numSeq[k] = -1
         elif t.upper() == "T":
             numSeq[k] = 1          
         else:
             pass
     return numSeq
            
def numMappingIntN(sq): 
#needs validation of output
     dob = ['T', 'C', 'A', 'G']
     length = len(sq)
     numSeq = np.zeros(length)
     for k in range(0, length):
         t = sq[k]
         tp = dob.index(t) + 1
         numSeq[k] = tp
     return numSeq
    
def numMappingInt(sq): 
#needs validation of output
     dob = ['T', 'C', 'A', 'G']
     length = len(sq)
     numSeq = np.zeros(length)
     for k in range(0, length):
         t = sq[k]
         tp = dob.index(t) 
         numSeq[k] = tp
     return numSeq
    
def numMappingEIIP(sq):
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
         t = sq[k]
         if t.upper() == "A":
             numSeq[k] = 0.1260
         elif t.upper() == "C":
             numSeq[k] = 0.1340
         elif t.upper() == "G":
             numSeq[k] = 0.0806
         elif t.upper() == "T":
             numSeq[k] = 0.1335        
         else:
             pass
    return numSeq
            
def numMappingAtomic(sq):
    length = len(sq)
    numSeq = np.zeros(length)
    for k in range(0, length):
         t = sq[k]
         if t.upper() == "A":
             numSeq[k] = 70
         elif t.upper() == "C":
             numSeq[k] = 58
         elif t.upper() == "G":
             numSeq[k] = 78
         elif t.upper() == "T":
             numSeq[k] = 66    
         else:
             pass
    return numSeq
         
def numMappingCodons(sq):
#needs validation of output
    length = len(sq) -1
    numSeq = np.zeros(length)
    codons = ['TTT','TTC','TTA','TTG','CTT','CTC','CTA','CTG','TCT','TCC','TCA','TCG','AGT','AGC','TAT','TAC',
              'TAA','TAG','TGA','TGT','TGC','TGG','CCT','CCC','CCA','CCG','CAT','CAC','CAA','CAG','CGT','CGC',
              'CGA','CGG','AGA','AGG','ATT','ATC','ATA','ATG','ACT','ACC','ACA','ACG','AAT','AAC','AAA','AAG',
              'GTT','GTC','GTA','GTG','GCT','GCC','GCA','GCG','GAT','GAC','GAA','GAG','GGT','GGC','GGA','GGG']

    for k in range(0, length):
        if k < length-1:
            t = sq[k:k+3]
        elif k == length-1:
            t = sq[k:k+2] + sq[0]
        else:
            t = sq[k] + sq[0:2]
        
        tp = codons.index(t)
        numSeq[k] = tp
    return numSeq

def num_mapping_Doublet(sq, alpha=0):
#needs validation of output
    length = len(sq) - 1
    numSeq = np.zeros(len(sq))
    doublet = ['AA','AT','TA','AG','TT','TG','AC','TC','GA','CA','GT','GG','CT','GC','CG','CC']
    kStrings = 2*alpha + 1
    for k in range(0, length):
        if alpha == 0:
            if k < length:
                t = sq[k:k+2]
            else:
                t = sq[k] + sq[0]
            tp = doublet.index(t) 
            numSeq[k] = tp
        else:
            loc = 0
            for index in range(k - alpha, k + alpha + 1):
                sPos = index
                ePos = sPos + 1
                if sPos < 1:
                    sPos = length + sPos
                elif sPos > length:
                    sPos = sPos - length
                elif ePos > length:
                    ePos = ePos - length
                elif ePos < 1:
                    ePos = ePos + length    
                         
                if sPos == length and ePos == 1:
                    t = sq[length] + sq[0]
                else:
                    t = sq[sPos:ePos+1]
                    
                loc = loc + doublet.index(t)
            tp = loc/kStrings
            numSeq[k] = tp
    
    return numSeq
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
