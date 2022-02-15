"""
@Daniel
"""
from MLDSP_core.cgr import PuPyCGR, oneDPuPyCGR, compute_cgr
from MLDSP_core.one_dimensional_num_mapping import *

# Dictionary order & names dependent for downstream execution
methods_list = {1: num_mapping_PP,
                2: num_mapping_Int,
                3: num_mapping_IntN,
                4: num_mapping_Real,
                5: num_mapping_Doublet,
                6: num_mapping_Codons,
                7: num_mapping_Atomic,
                8: num_mapping_EIIP,
                9: num_mapping_AT_CG,
                10: num_mapping_justA,
                11: num_mapping_justC,
                12: num_mapping_justG,
                13: num_mapping_justT,
                14: compute_cgr,
                15: PuPyCGR,
                16: oneDPuPyCGR}
