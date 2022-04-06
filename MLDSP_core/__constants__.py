"""
@Daniel
"""
from MLDSP_core.cgr import PuPyCGR, oneDPuPyCGR, compute_cgr
from MLDSP_core.one_dimensional_num_mapping import *

# Dictionary order & names dependent for downstream execution
methods_list = {'Purine/Pyrimidine': num_mapping_PP,
                'Integer': num_mapping_Int,
                'Integer natural': num_mapping_IntN,
                'Integer real': num_mapping_Real,
                'Doublet': num_mapping_Doublet,
                'Codons': num_mapping_Codons,
                'Atomic': num_mapping_Atomic,
                'EIIP': num_mapping_EIIP,
                'GC-AT': num_mapping_AT_CG,
                'Just As': num_mapping_justA,
                'Just Cs': num_mapping_justC,
                'Just Gs': num_mapping_justG,
                'Just Ts': num_mapping_justT,
                'Chaos Game Representation (CGR)': compute_cgr,
                'Pu/Py CGR': PuPyCGR,
                'Last row CGR': oneDPuPyCGR}

DEFAULT_METHOD = 'Chaos Game Representation (CGR)'
CGRS = ['Chaos Game Representation (CGR)', 'Pu/Py CGR', 'Last row CGR']
